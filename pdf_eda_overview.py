#!/usr/bin/env python3
"""
PDF Overview and Visualization Tool using spaCy-Layout (enhanced)

What's new in this version:
- Automatic language detection (EN/DE) and spaCy model switching with transformer preference.
- KeyBERT initialized once; language-aware stopwords.
- NER now includes frequency counts per entity and totals (total vs unique).
- Dashboard: truncates long keywords, POS labels include absolute counts + %, and a Word Cloud panel (keywords + NER).

Outputs:
- JSON files are saved in a 'NER' directory.
- Dashboard images (PNG) are saved in a 'dashboard' directory.
"""
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

# Optional dependency: wordcloud
try:
    from wordcloud import WordCloud

    _HAS_WORDCLOUD = True
except Exception:
    _HAS_WORDCLOUD = False

import spacy
from spacy_layout import spaCyLayout
from keybert import KeyBERT
from langdetect import detect
from spacy.lang.en.stop_words import STOP_WORDS as EN_STOPWORDS
from spacy.lang.de.stop_words import STOP_WORDS as DE_STOPWORDS

# sklearn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


class PDFOverviewAnalyzer:
    """
    Analyzes a PDF to generate a summary, content distribution, and NER,
    outputting results to JSON and a visual dashboard.
    """

    def __init__(self, output_dir_json="NER", output_dir_dashboard="dashboard", prefer_trf=False):
        """
        Initializes the analyzer and ensures output directories exist.

        Args:
            prefer_trf (bool): If True, attempt to load transformer models (en_core_web_trf / de_dep_news_trf)
                               when available. Falls back to small models if not installed.
        """
        self.prefer_trf = prefer_trf
        self.lang = "en"  # will be updated after first pass
        self.current_stopwords = list(EN_STOPWORDS)

        print("Loading initial spaCy model 'en_core_web_sm' (temporary, may switch after language detect)...")
        self.nlp = self._safe_load_spacy("en", prefer_trf=False)  # temp English to bootstrap
        self.layout_processor = spaCyLayout(self.nlp)
        print("✓ Initial model loaded. Will switch if document language differs.")

        # Initialize KeyBERT once
        self.kw_model = KeyBERT()

        # Create output directories
        self.json_dir = Path(output_dir_json)
        self.dashboard_dir = Path(output_dir_dashboard)
        self.json_dir.mkdir(exist_ok=True)
        self.dashboard_dir.mkdir(exist_ok=True)
        print(f"✓ Output will be saved to '{self.json_dir}/' and '{self.dashboard_dir}/'")

    # ------------------------ helpers ------------------------
    def _safe_load_spacy(self, lang: str, prefer_trf: bool = None):
        """Try loading a spaCy model for the given language with graceful fallbacks."""
        if prefer_trf is None:
            prefer_trf = self.prefer_trf
        try_order = []
        if lang == "de":
            if prefer_trf:
                try_order.append("de_dep_news_trf")
            try_order.append("de_core_news_sm")
        else:  # default to English
            if prefer_trf:
                try_order.append("en_core_web_trf")
            try_order.append("en_core_web_sm")

        last_err = None
        for model in try_order:
            try:
                print(f"→ Trying spaCy model '{model}'...")
                return spacy.load(model)
            except Exception as e:
                last_err = e
                print(f"  ⚠️  Could not load '{model}': {e}")
        # Final fallback
        print("  ❗ Falling back to 'en_core_web_sm'. Run 'python -m spacy download <model>' to install better models.")
        return spacy.load("en_core_web_sm")

    def _reload_pipeline_for_language(self, lang: str):
        """Recreate spaCy pipeline and layout processor for a detected language."""
        if lang not in ("en", "de"):
            lang = "en"
        self.lang = lang
        self.current_stopwords = list(DE_STOPWORDS) if lang == "de" else list(EN_STOPWORDS)
        self.nlp = self._safe_load_spacy(lang)
        self.layout_processor = spaCyLayout(self.nlp)
        print(f"✓ Switched pipeline to language='{lang}'")

    def _get_file_prefix(self, pdf_path: str) -> str:
        return Path(pdf_path).stem

    def _deduplicate_keywords(self, keywords):
        filtered = []
        for kw, score in keywords:
            is_duplicate = False
            for existing_kw, _ in filtered:
                if kw in existing_kw or existing_kw in kw:
                    if len(kw) > len(existing_kw):
                        filtered = [(k, s) for k, s in filtered if k != existing_kw]
                        filtered.append((kw, score))
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered.append((kw, score))
        return filtered[:10]

    def _truncate_label(self, s: str, max_len: int = 28) -> str:
        s = (s or "").strip()
        return s if len(s) <= max_len else (s[: max_len - 1] + "…")

    # ------------------------ analysis steps ------------------------
    def _generate_overview_and_keywords(self, text: str, top_n=15):
        """Extract keywords using KeyBERT with language-aware stopwords."""
        text = (text or "").strip()
        if not text:
            return "Could not determine the main topic (empty text).", []

        stop_words = self.current_stopwords if self.current_stopwords else "english"
        keywords = self.kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words=stop_words,
            top_n=top_n,
        )
        dedup = self._deduplicate_keywords(keywords)
        top_keywords = [kw for kw, _ in dedup]
        overview = f"The document primarily discusses topics related to: {', '.join(top_keywords)}."
        return overview, top_keywords

    def _analyze_content_distribution(self, doc):
        total_text_length = len(doc.text)
        table_text_length = 0
        if hasattr(doc._, 'tables') and doc._.tables:
            for table in doc._.tables:
                try:
                    table_text_length += len(table.text)
                except Exception:
                    pass
        if total_text_length == 0:
            return {
                "text_char_count": 0,
                "table_char_count": 0,
                "table_content_percentage": 0,
                "summary": "Document is empty or contains no extractable text.",
            }
        table_percentage = (table_text_length / total_text_length) * 100
        if table_percentage > 60:
            summary = f"The document is heavily table-based, with {table_percentage:.1f}% of its text content within tables."
        elif table_percentage > 30:
            summary = f"The document contains a significant amount of data in tables ({table_percentage:.1f}%)."
        else:
            summary = f"The document is primarily text-based, with only {table_percentage:.1f}% of its content in tables."
        return {
            "text_char_count": total_text_length - table_text_length,
            "table_char_count": table_text_length,
            "table_content_percentage": round(table_percentage, 2),
            "summary": summary,
        }

    def _extract_ner(self, doc):
        """Extract entities and include frequency counts per label."""
        entities_by_label = {}
        for ent in doc.ents:
            label = ent.label_
            entities_by_label.setdefault(label, []).append(ent.text)
        entities_with_counts = {}
        for label, items in entities_by_label.items():
            counts = Counter([t.strip() for t in items if t.strip()])
            entities_with_counts[label] = [
                {"text": text, "count": count} for text, count in counts.most_common()
            ]
        total_entities = sum(c["count"] for lst in entities_with_counts.values() for c in lst)
        unique_entities = sum(len(lst) for lst in entities_with_counts.values())
        return {
            "total_entities": total_entities,
            "unique_entities": unique_entities,
            "entities_by_type": entities_with_counts,
        }

    def _analyze_pos_distribution(self, doc):
        pos_counts = Counter([t.pos_ for t in doc if not t.is_punct and not t.is_space])
        total_tokens = sum(pos_counts.values())
        if total_tokens == 0:
            return {
                "noun_count": 0,
                "verb_count": 0,
                "other_count": 0,
                "noun_percentage": 0,
                "verb_percentage": 0,
                "other_percentage": 0,
                "total_tokens": 0,
            }
        noun_count = pos_counts.get('NOUN', 0) + pos_counts.get('PROPN', 0)
        verb_count = pos_counts.get('VERB', 0) + pos_counts.get('AUX', 0)
        other_count = total_tokens - noun_count - verb_count
        return {
            "noun_count": noun_count,
            "verb_count": verb_count,
            "other_count": other_count,
            "noun_percentage": round((noun_count / total_tokens) * 100, 1),
            "verb_percentage": round((verb_count / total_tokens) * 100, 1),
            "other_percentage": round((other_count / total_tokens) * 100, 1),
            "total_tokens": total_tokens,
        }

    def _extract_topics(self, text, n_topics=3, n_top_words=7):
        text = (text or "").strip()
        if not text:
            return []
        try:
            sentences = text.split('. ')
            meaningful = [s.strip() + '.' for s in sentences if len(s.split()) > 5]
            if len(meaningful) < 3:
                return []
            if len(meaningful) > 100:
                meaningful = meaningful[:100]
            vectorizer = TfidfVectorizer(
                stop_words=self.current_stopwords if self.current_stopwords else "english",
                max_df=0.8,
                min_df=2,
                max_features=1000,
                ngram_range=(1, 2),
            )
            tfidf = vectorizer.fit_transform(meaningful)
            feature_names = vectorizer.get_feature_names_out()
            actual_n_topics = min(n_topics, len(meaningful) // 3, 5)
            if actual_n_topics < 1:
                return []
            nmf = NMF(n_components=actual_n_topics, random_state=42, max_iter=200)
            nmf.fit(tfidf)
            topics = []
            for topic_idx, topic in enumerate(nmf.components_):
                top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
                topics.append({"topic_number": topic_idx + 1, "top_words": top_words})
            return topics
        except Exception as e:
            print(f"⚠️ Topic modeling failed: {e}")
            return []

    # ------------------------ visualization ------------------------
    def _create_dashboard(self, data: dict, output_path: str):
        """Generates and saves a visual dashboard summarizing the PDF analysis."""
        # Layout: 2 rows x 3 cols: Overview, POS, WordCloud \n Keywords, NER, (spacer)
        fig = plt.figure(figsize=(22, 14))
        gs = fig.add_gridspec(
            2, 3,
            height_ratios=[1, 1.2],
            width_ratios=[1.1, 1, 1],
            hspace=0.35, wspace=0.35, top=0.92, bottom=0.08, left=0.06, right=0.97
        )
        fig.suptitle(f"PDF Analysis Dashboard: {data['metadata']['filename']}", fontsize=20, fontweight='bold')

        # --- 1. Overview Panel ---
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        ax1.set_title("Document Overview", fontsize=16, pad=16, fontweight='bold')
        pos = data['pos_distribution']
        overview_lines = [
            f"Pages: {data['metadata']['page_count']}",
            f"Words: {data['metadata']['word_count']:,}",
            f"Tables: {data['metadata']['table_count']}",
            f"Language: {data['metadata'].get('language', 'en')}",
            f"Total Tokens: {pos['total_tokens']:,}",
            f"Total Entities: {data['named_entities']['total_entities']:,} (unique: {data['named_entities']['unique_entities']:,})",
            "",
            f"Content Distribution:",
            f"• Table content: {data['content_distribution']['table_content_percentage']:.1f}%",
        ]
        ax1.text(
            0.05, 0.85, '\n'.join(overview_lines), va='top', ha='left', wrap=True, fontsize=10,
            transform=ax1.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", alpha=0.3),
        )

        # --- 2. POS Distribution (Donut with counts + %) ---
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title("Parts of Speech Distribution", fontsize=16, pad=16, fontweight='bold')
        noun_c, verb_c, other_c = pos['noun_count'], pos['verb_count'], pos['other_count']
        labels = [
            f"Nouns ({noun_c})",
            f"Verbs ({verb_c})",
            f"Other POS ({other_c})",
        ]
        sizes = [noun_c, verb_c, other_c]
        wedges, texts, autotexts = ax2.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
            textprops={'fontsize': 9},
            pctdistance=0.75,
            labeldistance=1.1,
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        for text in texts:
            text.set_fontsize(9)
            text.set_fontweight('bold')
        ax2.axis('equal')
        ax2.set_xlim(-1.2, 1.2)
        ax2.set_ylim(-1.2, 1.2)

        # --- 3. Word Cloud (keywords + top NER) ---
        ax_wc = fig.add_subplot(gs[0, 2])
        ax_wc.set_title("Word Cloud (Keywords + NER)", fontsize=16, pad=16, fontweight='bold')
        if _HAS_WORDCLOUD:
            freq = Counter()
            # weight keywords higher
            for kw in data['overview']['top_keywords']:
                if kw:
                    freq[kw] += 3
            # add top NER per label
            for label, items in data['named_entities']['entities_by_type'].items():
                for rec in items[:15]:  # top per label
                    freq[rec['text']] += rec['count']
            if freq:
                wc = WordCloud(width=900, height=600, background_color='white')
                wc.generate_from_frequencies(freq)
                ax_wc.imshow(wc, interpolation='bilinear')
                ax_wc.axis('off')
            else:
                ax_wc.text(0.5, 0.5, "Insufficient data for word cloud", ha='center', va='center')
        else:
            ax_wc.text(0.5, 0.5, "Install 'wordcloud' to enable this panel", ha='center', va='center')
            ax_wc.axis('off')

        # --- 4. Top Keywords Panel (truncated to avoid overflow) ---
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.axis('off')
        ax3.set_title("Top 10 Keywords", fontsize=16, pad=16, fontweight='bold')
        keywords_text = ""
        for i, kw in enumerate(data['overview']['top_keywords'][:10], 1):
            keywords_text += f"{i:2d}. {self._truncate_label(kw)}\n"
        ax3.text(
            0.05, 0.9, keywords_text, va='top', ha='left', fontsize=11,
            transform=ax3.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.3),
        )

        # --- 5. NER Breakdown (Bar Chart) using counts ---
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_title("Named Entity Recognition (by label: unique counts)", fontsize=16, pad=16, fontweight='bold')
        ner_data = data['named_entities']['entities_by_type']
        if ner_data:
            ner_counts = {label: len(entities) for label, entities in ner_data.items()}
            sorted_ner = sorted(ner_counts.items(), key=lambda x: x[1], reverse=True)[:15]
            labels_ner = [item[0] for item in sorted_ner]
            counts = [item[1] for item in sorted_ner]
            bars = ax4.barh(labels_ner, counts)
            ax4.set_xlabel("Number of Unique Entities", fontsize=12)
            ax4.grid(axis='x', alpha=0.3)
            for bar, count in zip(bars, counts):
                ax4.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2, str(count),
                         va='center', fontsize=9)
        else:
            ax4.text(0.5, 0.5, "No entities found", ha='center', va='center', fontsize=12, transform=ax4.transAxes)
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)

        # --- 6. Spacer or future panel ---
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')

        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✓ Dashboard saved to: {output_path}")

    # ------------------------ public API ------------------------
    def analyze(self, pdf_path: str):
        print("\n" + "=" * 50)
        print(f"🚀 Starting analysis for: {Path(pdf_path).name}")
        print("=" * 50)
        if not Path(pdf_path).exists():
            print(f"❌ Error: File not found at '{pdf_path}'")
            return None

        # 1. First pass: process PDF, then detect language on extracted text
        print("1. Processing PDF with spaCy-Layout (initial pass)...")
        try:
            doc_first = self.layout_processor(pdf_path)
            doc_first = self.nlp(doc_first)
        except Exception as e:
            print(f"❌ Failed to process PDF. Error: {e}")
            return None

        # 2. Detect language and, if needed, re-run with appropriate pipeline
        try:
            detected = detect(doc_first.text) if doc_first and doc_first.text.strip() else "en"
            print(f"→ Detected language: {detected}")
        except Exception as e:
            print(f"⚠️ Language detection failed ({e}); defaulting to 'en'")
            detected = "en"

        if detected in ("en", "de") and detected != self.lang:
            print("2a. Switching pipeline based on detected language and re-processing…")
            self._reload_pipeline_for_language(detected)
            try:
                doc = self.layout_processor(pdf_path)
                doc = self.nlp(doc)
            except Exception as e:
                print(f"❌ Failed to re-process PDF after language switch. Error: {e}")
                doc = doc_first
        else:
            doc = doc_first
            self.lang = detected if detected in ("en", "de") else "en"
            self.current_stopwords = list(DE_STOPWORDS) if self.lang == "de" else list(EN_STOPWORDS)

        file_prefix = self._get_file_prefix(pdf_path)

        # 3. Run analysis modules
        print("3. Generating overview and analyzing content…")
        overview, keywords = self._generate_overview_and_keywords(doc.text)
        content_dist = self._analyze_content_distribution(doc)
        ner_results = self._extract_ner(doc)
        pos_distribution = self._analyze_pos_distribution(doc)
        print("4. Extracting topics…")
        topics = self._extract_topics(doc.text)

        results = {
            "metadata": {
                "filename": Path(pdf_path).name,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "page_count": len(doc._.layout.pages) if hasattr(doc._.layout, 'pages') else 'N/A',
                "word_count": len([t for t in doc if not t.is_space]),
                "table_count": len(doc._.tables) if hasattr(doc._, 'tables') else 0,
                "language": self.lang,
                "spaCy_model": getattr(self.nlp, 'meta', {}).get('name', 'unknown'),
            },
            "overview": {
                "summary": overview,
                "top_keywords": keywords,
                "topics": topics,
            },
            "content_distribution": content_dist,
            "named_entities": ner_results,
            "pos_distribution": pos_distribution,
        }

        # 5. Save JSON
        print("5. Saving JSON analysis…")
        json_output_path = self.json_dir / f"{file_prefix}_analysis.json"
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"✓ JSON results saved to: {json_output_path}")

        # 6. Create and save dashboard
        print("6. Creating visual dashboard…")
        dashboard_output_path = self.dashboard_dir / f"{file_prefix}_dashboard.png"
        self._create_dashboard(results, dashboard_output_path)

        print(f"\n🎉 Analysis complete for {Path(pdf_path).name}!")
        return results


def main():
    analyzer = PDFOverviewAnalyzer(prefer_trf=True)

    # Add your PDF paths here
    pdf_files_to_process = [
        # Example:
        "/Users/preethisivakumar/Documents/spacelyout/sample_pdf/gesamtes LV.pdf"
    ]

    for pdf_file in pdf_files_to_process:
        analyzer.analyze(pdf_file)


if __name__ == "__main__":
    main()
