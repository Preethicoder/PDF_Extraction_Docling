#!/usr/bin/env python3
"""
Fixed PDF Analyzer with JSON Output Format and Table Merging - Filename Prefixed Outputs

Fixed error handling and added debugging for table extraction issues.
Now creates separate files for each PDF with filename prefix.

Requirements:
pip install spacy spacy-layout pandas matplotlib seaborn fitz PyMuPDF
python -m spacy download en_core_web_sm
"""

import spacy
from spacy_layout import spaCyLayout
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import os
import json
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional, Union
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io
from difflib import SequenceMatcher


class SmartTableMerger:
    """Smart table merger that maintains your JSON format"""

    def __init__(
            self,
            column_similarity_threshold: float = 0.85,
            page_proximity_limit: int = 2,
            header_confidence_threshold: float = 0.7,
            header_keywords: List[str] = None,
            data_keywords: List[str] = None
    ):
        self.column_similarity_threshold = column_similarity_threshold
        self.page_proximity_limit = page_proximity_limit
        self.header_confidence_threshold = header_confidence_threshold


        # Configurable keywords - now without any hardcoded defaults
        self.header_keywords = header_keywords or []
        self.data_keywords = data_keywords or []

    def is_header_table(self, table_info: Dict[str, Any]) -> bool:
        """Detect header tables using structural analysis rather than hardcoded keywords"""
        try:
            # Single row tables are likely headers
            if table_info.get('rows', 0) == 1:
                return True

            # Tables with very few rows (2-3) and mostly text content could be headers
            if table_info.get('rows', 0) <= 3:
                rows_data = table_info.get('rows_data', [])
                if not rows_data:
                    return False

                # Check if most cells contain non-numeric data (text-heavy = likely header)
                text_cell_count = 0
                total_cells = 0

                for row in rows_data:
                    if not isinstance(row, (list, tuple)):
                        continue
                    for cell in row:
                        total_cells += 1
                        cell_str = str(cell).strip()
                        # Consider it text if it's not purely numeric or empty
                        if cell_str and not (cell_str.replace('.', '').replace('-', '').replace(',', '').isdigit()):
                            text_cell_count += 1

                if total_cells > 0:
                    text_ratio = text_cell_count / total_cells
                    # If more than 70% text content, likely a header table
                    return text_ratio > 0.7

            # Optionally check against provided header keywords if any
            if self.header_keywords:
                column_names = table_info.get('column_names', [])
                if column_names:
                    text_content = ' '.join(str(col) for col in column_names).upper()
                    return any(keyword.upper() in text_content for keyword in self.header_keywords)

            return False
        except Exception as e:
            print(f"Warning: Error in header detection: {e}")
            return False

    def is_data_table(self, table_info: Dict[str, Any]) -> bool:
        """Detect data tables using structural analysis and optional keywords"""
        try:
            # Must have more than 1 row to be considered data
            if table_info.get('rows', 0) <= 1:
                return False

            rows_data = table_info.get('rows_data', [])
            if not rows_data or len(rows_data) < 2:
                return False

            # Analyze data structure - look for mixed content (headers + data)
            if len(rows_data) >= 2:
                # Check if we have a header row followed by data rows
                header_row = rows_data[0]
                data_rows = rows_data[1:]

                if not isinstance(header_row, (list, tuple)) or not header_row:
                    return False

                # Header should be mostly text
                header_text_count = 0
                for cell in header_row:
                    cell_str = str(cell).strip()
                    if cell_str and not (cell_str.replace('.', '').replace('-', '').replace(',', '').isdigit()):
                        header_text_count += 1

                header_text_ratio = header_text_count / len(header_row) if len(header_row) > 0 else 0

                # Data rows should have some numeric content
                numeric_cells = 0
                total_data_cells = 0

                for row in data_rows:
                    if not isinstance(row, (list, tuple)):
                        continue
                    for cell in row:
                        total_data_cells += 1
                        cell_str = str(cell).strip()
                        # Check if cell contains numbers or typical data patterns
                        if (cell_str.replace('.', '').replace('-', '').replace(',', '').isdigit() or
                                any(char.isdigit() for char in cell_str)):
                            numeric_cells += 1

                data_numeric_ratio = numeric_cells / total_data_cells if total_data_cells > 0 else 0

                # Consider it a data table if header is mostly text and data has some numbers
                if header_text_ratio > 0.5 and data_numeric_ratio > 0.2:
                    return True

            # Optionally check against provided data keywords if any
            if self.data_keywords:
                column_names = table_info.get('column_names', [])
                if column_names:
                    text_content = ' '.join(str(col) for col in column_names).lower()
                    return any(keyword.lower() in text_content for keyword in self.data_keywords)

            return False
        except Exception as e:
            print(f"Warning: Error in data table detection: {e}")
            return False

    def calculate_column_similarity(self, cols1: List[str], cols2: List[str]) -> float:
        """Calculate similarity between two column sets"""
        if not cols1 or not cols2 or len(cols1) != len(cols2):
            return 0.0

        try:
            # Clean and normalize column names
            clean_cols1 = [str(col).strip().lower() for col in cols1]
            clean_cols2 = [str(col).strip().lower() for col in cols2]

            # Calculate exact matches
            exact_matches = sum(1 for c1, c2 in zip(clean_cols1, clean_cols2) if c1 == c2)
            exact_similarity = exact_matches / len(cols1) if len(cols1) > 0 else 0.0

            # If not perfect match, try fuzzy matching
            if exact_similarity < 1.0:
                fuzzy_scores = []
                for c1, c2 in zip(clean_cols1, clean_cols2):
                    try:
                        score = SequenceMatcher(None, c1, c2).ratio()
                        fuzzy_scores.append(score)
                    except Exception:
                        fuzzy_scores.append(0.0)

                fuzzy_similarity = sum(fuzzy_scores) / len(fuzzy_scores) if len(fuzzy_scores) > 0 else 0.0
                return max(exact_similarity, fuzzy_similarity)

            return exact_similarity
        except Exception as e:
            print(f"Warning: Error calculating column similarity: {e}")
            return 0.0

    def should_merge_tables(self, table1: Dict, table2: Dict) -> bool:
        """Determine if two tables should be merged"""
        # Both must be data tables
        if not (self.is_data_table(table1) and self.is_data_table(table2)):
            return False

        # Check column similarity
        cols1 = table1.get('column_names', [])
        cols2 = table2.get('column_names', [])
        similarity = self.calculate_column_similarity(cols1, cols2)

        if similarity < self.column_similarity_threshold:
            return False

        # Check page proximity (if page info available)
        page1 = table1.get('page', 1)
        page2 = table2.get('page', 1)

        if abs(page2 - page1) > self.page_proximity_limit:
            return False

        # Check if second table looks like continuation (no strong header)
        if self._has_strong_header(table2):
            return False

        return True

    def _has_strong_header(self, table_info: Dict) -> bool:
        """Check if table has a strong header row"""
        rows_data = table_info.get('rows_data', [])
        if not rows_data or len(rows_data) < 2:
            return False

        first_row = rows_data[0]
        second_row = rows_data[1]

        # Check if first row is mostly text and second row has more data-like content
        first_row_text_count = sum(1 for val in first_row
                                   if isinstance(val, str) and not str(val).replace('.', '').replace('-', '').isdigit())

        second_row_numeric_count = sum(1 for val in second_row
                                       if isinstance(val, (int, float)) or
                                       (isinstance(val, str) and str(val).replace('.', '').replace('-', '').isdigit()))

        first_row_text_ratio = first_row_text_count / len(first_row) if first_row else 0
        second_row_numeric_ratio = second_row_numeric_count / len(second_row) if second_row else 0

        return first_row_text_ratio > 0.6 and second_row_numeric_ratio > 0.3

    def merge_table_group(self, table_group: List[Dict]) -> Dict:
        """Merge a group of tables maintaining JSON format"""
        if not table_group:
            return {}

        if len(table_group) == 1:
            return table_group[0]

        # Use first table as base
        base_table = table_group[0].copy()
        base_rows_data = base_table.get('rows_data', [])

        # Merge additional tables
        for table in table_group[1:]:
            additional_rows = table.get('rows_data', [])

            # Skip header-like rows in continuation tables
            filtered_rows = self._filter_header_rows(additional_rows, base_table['column_names'])
            base_rows_data.extend(filtered_rows)

        # Update merged table info
        base_table['rows_data'] = base_rows_data
        base_table['rows'] = len(base_rows_data)
        base_table['merged_from'] = [t['table_id'] for t in table_group]
        base_table['pages_spanned'] = list(set(t.get('page', 1) for t in table_group))

        return base_table

    def _filter_header_rows(self, rows_data: List[List], reference_columns: List[str]) -> List[List]:
        """Filter out rows that look like headers"""
        if not rows_data or not reference_columns:
            return rows_data

        try:
            filtered_rows = []
            ref_cols_lower = [str(col).strip().lower() for col in reference_columns]

            for row in rows_data:
                if not isinstance(row, (list, tuple)):
                    continue

                row_lower = [str(val).strip().lower() for val in row]

                # Check similarity to reference columns (only if lengths match)
                if len(row_lower) == len(ref_cols_lower):
                    similarity_count = sum(1 for val, ref in zip(row_lower, ref_cols_lower) if val == ref)
                    similarity_ratio = similarity_count / len(reference_columns) if len(reference_columns) > 0 else 0

                    # Keep row if it's not too similar to header
                    if similarity_ratio < 0.7:
                        filtered_rows.append(row)
                else:
                    # Different lengths, probably not a header - keep it
                    filtered_rows.append(row)

            return filtered_rows
        except Exception as e:
            print(f"Warning: Error filtering header rows: {e}")
            return rows_data

    def process_tables(self, tables_analysis: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Process and merge tables while maintaining JSON format"""
        # Initialize merge_info with default structure
        merge_info = {
            'original_table_count': len(tables_analysis),
            'header_tables': 0,
            'data_tables': 0,
            'other_tables': 0,
            'merge_groups': [],
            'final_table_count': 0
        }

        if not tables_analysis:
            return [], merge_info

        # Separate header and data tables
        header_tables = []
        data_tables = []
        other_tables = []

        for table in tables_analysis:
            if self.is_header_table(table):
                header_tables.append(table)
            elif self.is_data_table(table):
                data_tables.append(table)
            else:
                other_tables.append(table)

        # Update counts
        merge_info.update({
            'header_tables': len(header_tables),
            'data_tables': len(data_tables),
            'other_tables': len(other_tables)
        })

        # Group data tables for merging
        merged_tables = []
        processed_indices = set()

        for i, table in enumerate(data_tables):
            if i in processed_indices:
                continue

            # Start new merge group
            merge_group = [table]
            group_info = {
                'primary_table_id': table['table_id'],
                'merged_table_ids': [table['table_id']],
                'pages': [table.get('page', 1)]
            }

            # Find tables to merge with this one
            for j, candidate in enumerate(data_tables[i + 1:], i + 1):
                if j in processed_indices:
                    continue

                if self.should_merge_tables(table, candidate):
                    merge_group.append(candidate)
                    group_info['merged_table_ids'].append(candidate['table_id'])
                    group_info['pages'].append(candidate.get('page', 1))
                    processed_indices.add(j)

            # Merge the group
            merged_table = self.merge_table_group(merge_group)
            group_info['final_rows'] = merged_table['rows']
            group_info['tables_merged_count'] = len(merge_group)

            merged_tables.append(merged_table)
            merge_info['merge_groups'].append(group_info)
            processed_indices.add(i)

        # Combine all tables (headers + merged data + others)
        all_processed_tables = header_tables + merged_tables + other_tables

        # Renumber table IDs
        for i, table in enumerate(all_processed_tables, 1):
            table['table_id'] = i

        merge_info['final_table_count'] = len(all_processed_tables)

        return all_processed_tables, merge_info


class JSONFormatPDFAnalyzer:
    """PDF Analyzer that outputs in your specific JSON format with table merging"""

    def __init__(self, output_dir="json_eda", header_keywords=None, data_keywords=None):
        """Initialize the analyzer with configurable keywords"""
        print("Loading spaCy model...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.layout_processor = spaCyLayout(self.nlp)
        except Exception as e:
            print(f"❌ Error loading spaCy model: {e}")
            print("Try running: python -m spacy download en_core_web_sm")
            raise

        self.doc = None
        self.output_dir = output_dir
        self.table_merger = SmartTableMerger(
            header_keywords=header_keywords,
            data_keywords=data_keywords
        )

        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"✓ Created output directory: {self.output_dir}")

        print(f"✓ Table detection configured:")
        print(f"  - Header keywords: {header_keywords or 'None (using structural analysis)'}")
        print(f"  - Data keywords: {data_keywords or 'Default multilingual set'}")

    def get_file_prefix(self, pdf_path: str) -> str:
        """Extract filename without extension to use as prefix"""
        try:
            # Handle both string paths and Path objects
            path_obj = Path(pdf_path)
            # Clean the filename to be filesystem-safe
            filename = path_obj.stem
            # Remove any problematic characters
            safe_filename = "".join(c for c in filename if c.isalnum() or c in ('-', '_', '.'))
            return safe_filename if safe_filename else "document"
        except Exception as e:
            print(f"Warning: Error extracting filename: {e}")
            return "document"

    def load_pdf(self, pdf_path):
        """Load and process PDF with better error handling"""
        print(f"Processing: {pdf_path}")

        # Check if file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            self.doc = self.layout_processor(pdf_path)
            self.doc = self.nlp(self.doc)
            print("✓ PDF loaded successfully!")

            # Debug: Check what was loaded
            print(f"  - Document length: {len(self.doc)} tokens")
            print(f"  - Has layout info: {hasattr(self.doc._, 'layout')}")
            print(f"  - Has tables attr: {hasattr(self.doc._, 'tables')}")

            if hasattr(self.doc._, 'tables'):
                print(f"  - Number of tables: {len(self.doc._.tables)}")
            else:
                print("  - No tables attribute found")

        except Exception as e:
            print(f"❌ Error loading PDF: {e}")
            raise

    def basic_stats(self):
        """Get basic document statistics"""
        if not self.doc:
            return {}

        try:
            stats = {
                'Total Tokens': len(self.doc),
                'Total Characters': len(self.doc.text),
                'Total Sentences': len(list(self.doc.sents)),
                'Has Layout Info': hasattr(self.doc._, 'layout'),
                'Has Tables': hasattr(self.doc._, 'tables') and len(self.doc._.tables) > 0,
                'Layout Spans': len(self.doc.spans.get('layout', []))
            }

            # Page information
            if hasattr(self.doc._, 'layout') and self.doc._.layout:
                if hasattr(self.doc._.layout, 'pages'):
                    stats['Number of Pages'] = len(self.doc._.layout.pages)

            return stats
        except Exception as e:
            print(f"Warning: Error calculating basic stats: {e}")
            return {
                'Total Tokens': 0,
                'Total Characters': 0,
                'Total Sentences': 0,
                'Has Layout Info': False,
                'Has Tables': False,
                'Layout Spans': 0
            }

    def analyze_layout(self):
        """Analyze document layout structure"""
        if not self.doc or 'layout' not in self.doc.spans:
            return {}

        layout_spans = self.doc.spans['layout']
        span_types = Counter(span.label_ for span in layout_spans)

        return span_types

    def analyze_text_content(self):
        """Analyze text content and extract full text"""
        if not self.doc:
            return {}

        try:
            total_tokens = len([t for t in self.doc if not t.is_space])

            # Extract full text content
            full_text = self.doc.text

            # Extract text by pages if available
            pages_text = []
            try:
                if hasattr(self.doc._, 'layout') and self.doc._.layout and hasattr(self.doc._.layout, 'pages'):
                    for i, page in enumerate(self.doc._.layout.pages, 1):
                        page_start = getattr(page, 'start', 0)
                        page_end = getattr(page, 'end', len(self.doc.text))

                        # Ensure valid bounds
                        page_start = max(0, min(page_start, len(self.doc.text)))
                        page_end = max(page_start, min(page_end, len(self.doc.text)))

                        page_text = self.doc.text[page_start:page_end].strip()

                        pages_text.append({
                            'page_number': i,
                            'text': page_text,
                            'character_count': len(page_text),
                            'word_count': len(page_text.split()) if page_text else 0
                        })
            except Exception as e:
                print(f"Warning: Error extracting page text: {e}")

            # Extract sentences
            sentences = []
            try:
                sentences = [sent.text.strip() for sent in self.doc.sents if sent.text.strip()]
            except Exception as e:
                print(f"Warning: Error extracting sentences: {e}")

            # Extract paragraphs (split by double newlines)
            paragraphs = []
            try:
                paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip()]
            except Exception as e:
                print(f"Warning: Error extracting paragraphs: {e}")

            return {
                'total_tokens': total_tokens,
                'full_text': full_text,
                'character_count': len(full_text),
                'word_count': len(full_text.split()) if full_text else 0,
                'sentence_count': len(sentences),
                'paragraph_count': len(paragraphs),
                'sentences': sentences,
                'paragraphs': paragraphs,
                'pages_text': pages_text
            }
        except Exception as e:
            print(f"Warning: Error analyzing text content: {e}")
            return {
                'total_tokens': 0,
                'full_text': '',
                'character_count': 0,
                'word_count': 0,
                'sentence_count': 0,
                'paragraph_count': 0,
                'sentences': [],
                'paragraphs': [],
                'pages_text': []
            }

    def extract_table_data(self, table_obj) -> Tuple[List[List[Any]], str]:
        """Extract all row data from a table object and its raw text"""
        try:
            # Extract raw text from the table span
            table_text = table_obj.text.strip() if hasattr(table_obj, 'text') else ""

            if not hasattr(table_obj._, 'data') or table_obj._.data is None:
                print(f"  - Table has no data attribute or data is None")
                # Return empty data but include the raw text
                return [], table_text

            df = table_obj._.data
            if df.empty:
                print(f"  - Table DataFrame is empty")
                return [], table_text

            # Convert DataFrame to list of lists (including header as first row)
            rows_data = []

            # Add header row
            header_row = [str(col) for col in df.columns]
            rows_data.append(header_row)

            # Add data rows
            for _, row in df.iterrows():
                data_row = []
                for val in row:
                    if pd.isna(val):
                        data_row.append("")
                    else:
                        data_row.append(str(val))
                rows_data.append(data_row)

            print(f"  - Successfully extracted {len(rows_data)} rows, {len(header_row)} columns")
            return rows_data, table_text

        except Exception as e:
            print(f"  - Error extracting table data: {e}")
            # Still try to get raw text even if structured data fails
            table_text = table_obj.text.strip() if hasattr(table_obj, 'text') else ""
            return [], table_text

    def analyze_tables_with_data(self):
        """Analyze tables and extract complete data in your JSON format"""
        if not self.doc:
            print("❌ No document loaded")
            return []

        if not hasattr(self.doc._, 'tables'):
            print("❌ Document has no tables attribute")
            return []

        tables = self.doc._.tables
        print(f"📋 Found {len(tables)} table objects")

        if len(tables) == 0:
            print("ℹ️  No tables detected in document")
            print("   This could be because:")
            print("   - The PDF doesn't contain tables")
            print("   - Tables are images/scanned content")
            print("   - spacy-layout couldn't detect table structure")
            return []

        table_analysis = []

        for i, table in enumerate(tables):
            print(f"  Processing table {i + 1}...")

            # Extract table data and raw text
            rows_data, table_raw_text = self.extract_table_data(table)

            table_info = {
                'table_id': i + 1,
                'position': f"Tokens {table.start}-{table.end}",
                'rows': len(rows_data),
                'columns': len(rows_data[0]) if rows_data else 0,
                'column_names': rows_data[0] if rows_data else [],
                'rows_data': rows_data,
                'raw_text': table_raw_text,
                'text_length': len(table_raw_text)
            }

            # Ensure column_names is always a list of strings
            if table_info['column_names']:
                table_info['column_names'] = [str(col) for col in table_info['column_names']]

            # Add page information if available
            if hasattr(table._, 'layout') and hasattr(table._.layout, 'page'):
                table_info['page'] = table._.layout.page
            else:
                table_info['page'] = 1

            # Add bounding box information if available
            if hasattr(table._, 'layout'):
                layout = table._.layout
                if hasattr(layout, 'bbox'):
                    table_info['bbox'] = {
                        'x0': layout.bbox.x0,
                        'y0': layout.bbox.y0,
                        'x1': layout.bbox.x1,
                        'y1': layout.bbox.y1
                    }

            table_analysis.append(table_info)

        return table_analysis

    def run_complete_analysis_with_merging(self, pdf_path):
        """Run complete analysis with table merging in your JSON format"""
        print("🔍 Starting JSON Format PDF Analysis with Table Merging")
        print("=" * 60)

        try:
            # Load PDF
            self.load_pdf(pdf_path)

            # Get file prefix for output naming
            file_prefix = self.get_file_prefix(pdf_path)
            print(f"📝 Using file prefix: {file_prefix}")

            # Run basic analyses
            basic_stats = self.basic_stats()
            layout_analysis = self.analyze_layout()
            content_analysis = self.analyze_text_content()

            # Extract tables with complete data
            raw_table_analysis = self.analyze_tables_with_data()

            print(f"📋 Raw tables extracted: {len(raw_table_analysis)}")

            # Process tables with merging
            processed_tables, merge_info = self.table_merger.process_tables(raw_table_analysis)

            print(f"📊 Tables after processing: {len(processed_tables)}")
            print(f"🔄 Merge groups created: {len(merge_info.get('merge_groups', []))}")

            # Create final results in your format
            results = {
                'basic_stats': basic_stats,
                'layout_analysis': dict(layout_analysis),
                'content_analysis': content_analysis,
                'table_analysis': processed_tables,
                'merge_info': merge_info
            }

            # Save complete results with filename prefix
            results_path = os.path.join(self.output_dir, f"{file_prefix}_complete_analysis_with_merging.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)

            print(f"✅ Complete analysis saved to: {results_path}")

            # Also save just the table analysis in your exact format
            table_only_results = {
                'basic_stats': basic_stats,
                'layout_analysis': dict(layout_analysis),
                'content_analysis': content_analysis,
                'table_analysis': processed_tables
            }

            table_results_path = os.path.join(self.output_dir, f"{file_prefix}_table_analysis_merged.json")
            with open(table_results_path, 'w', encoding='utf-8') as f:
                json.dump(table_only_results, f, indent=2, ensure_ascii=False, default=str)

            print(f"✅ Table analysis (your format) saved to: {table_results_path}")

            # Print summary
            self.print_analysis_summary(pdf_path,results, file_prefix)

            return results

        except Exception as e:
            print(f"❌ Error during analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def print_analysis_summary(self, pdf_path,results, file_prefix):
        """Print a summary of the analysis"""
        print(f"\n📊 ANALYSIS SUMMARY FOR: {file_prefix}")
        print("=" * 50)

        basic_stats = results['basic_stats']
        content_analysis = results['content_analysis']

        print(f"Pages: {basic_stats.get('Number of Pages', 'Unknown')}")
        print(f"Total Tokens: {basic_stats.get('Total Tokens', 0)}")
        print(f"Character Count: {content_analysis.get('character_count', 0)}")
        print(f"Word Count: {content_analysis.get('word_count', 0)}")
        print(f"Sentence Count: {content_analysis.get('sentence_count', 0)}")
        print(f"Paragraph Count: {content_analysis.get('paragraph_count', 0)}")
        print(f"Total Tables: {len(results['table_analysis'])}")

        merge_info = results['merge_info']
        print(f"Original Tables: {merge_info['original_table_count']}")
        print(f"Header Tables: {merge_info['header_tables']}")
        print(f"Data Tables: {merge_info['data_tables']}")
        print(f"Merge Groups: {len(merge_info.get('merge_groups', []))}")

        if results['table_analysis']:
            print(f"\n📋 TABLE DETAILS:")
            for table in results['table_analysis']:
                table_type = "HEADER" if len(table.get('merged_from', [])) == 0 and table['rows'] == 1 else "DATA"
                merged_info = f" (merged from tables {table.get('merged_from', [])})" if 'merged_from' in table else ""

                print(
                    f"  Table {table['table_id']}: {table['rows']} rows × {table['columns']} columns [{table_type}]{merged_info}")
                if table['column_names']:
                    # Ensure column names are strings and handle nested lists
                    col_names = table['column_names']
                    if col_names and isinstance(col_names[0], list):
                        # If first element is a list, flatten it
                        col_names = [str(item) for sublist in col_names for item in sublist]
                    else:
                        col_names = [str(col) for col in col_names]

                    display_cols = col_names[:3]
                    print(
                        f"    Columns: {', '.join(display_cols)}{'...' if len(col_names) > 3 else ''}")
                print(f"    Raw text length: {table.get('text_length', 0)} characters")
        else:
            print("\n📋 No tables found in the document")
            print("   Possible reasons:")
            print("   - Document contains no structured tables")
            print("   - Tables are embedded as images")
            print("   - Text-based tables not recognized by spacy-layout")

        # Show text content summary
        if content_analysis.get('pages_text'):
            print(f"\n📄 TEXT CONTENT BY PAGE:")
            for page_info in content_analysis['pages_text'][:3]:  # Show first 3 pages
                print(
                    f"  Page {page_info['page_number']}: {page_info['word_count']} words, {page_info['character_count']} chars")
            if len(content_analysis['pages_text']) > 3:
                print(f"  ... and {len(content_analysis['pages_text']) - 3} more pages")

        print(f"\n📁 Files generated for {file_prefix}:")
        print(f"  - {file_prefix}_complete_analysis_with_merging.json (with merge info)")
        print(f"  - {file_prefix}_table_analysis_merged.json (your exact format)")

    # Inside your PDFAnalyzer class

    def create_eda_dashboard(self, pdf_path,analysis_results: dict, output_dir: str):
        """
        Generates a visual EDA dashboard of the PDF document.
        """
        if not analysis_results or not analysis_results.get('basic_stats'):
            print("No analysis results available to create a dashboard.")
            return

        # Create directory for visualizations
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        dashboard_filename = f"{pdf_filename}_eda_dashboard.png"
        dashboard_pathdir = os.path.join(viz_dir, dashboard_filename)
        # Prepare data for plotting
        basic_stats = analysis_results['basic_stats']
        layout_counts = analysis_results['layout_analysis']

        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('PDF Exploratory Data Analysis Dashboard', fontsize=18, fontweight='bold')

        # --- Panel 1: Document Overview ---
        ax1 = plt.subplot(2, 2, 1)
        ax1.axis('off')  # Hide axis for text
        overview_text = (
            f"📄 **Document Overview**\n"
            f"-----------------------\n"
            f"• **Number of Pages:** {basic_stats.get('Number of Pages', 'N/A')}\n"
            f"• **Total Words:** {analysis_results['content_analysis'].get('word_count', 'N/A')}\n"
            f"• **Total Characters:** {basic_stats.get('Total Characters', 'N/A')}\n"
            f"• **Total Tables:** {basic_stats.get('Has Tables', 'N/A')}\n"
        )
        ax1.text(0.5, 0.5, overview_text, ha='center', va='center', fontsize=12, family='monospace')

        # --- Panel 2: Content Breakdown (Pie Chart) ---
        ax2 = plt.subplot(2, 2, 2)
        labels = list(layout_counts.keys())
        sizes = list(layout_counts.values())
        ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
        ax2.set_title('Content Type Breakdown', fontsize=14)

        # --- Panel 3: Tables Found (Bar Chart) ---
        ax3 = plt.subplot(2, 2, 3)
        if basic_stats.get('Has Tables'):
            table_counts = [
                analysis_results['merge_info']['final_table_count'],
                analysis_results['merge_info']['original_table_count']
            ]
            labels = ['Final Tables', 'Original Tables']
            sns.barplot(x=labels, y=table_counts, ax=ax3, palette='viridis')
            ax3.set_title('Table Analysis', fontsize=14)
            ax3.set_ylabel('Count')
        else:
            ax3.text(0.5, 0.5, 'No Tables Found', ha='center', va='center', fontsize=12)
            ax3.axis('off')

        # --- Panel 4: Text vs. Tables ---
        ax4 = plt.subplot(2, 2, 4)
        # Placeholder for a more advanced plot, for now, let's use a simple bar chart
        content_types = ['Text', 'Tables']
        word_count = analysis_results['content_analysis'].get('word_count', 0)
        table_words = sum([len(t['raw_text'].split()) for t in analysis_results['table_analysis'] if t.get('raw_text')])
        counts = [word_count - table_words, table_words]
        sns.barplot(x=content_types, y=counts, ax=ax4, palette='rocket')
        ax4.set_title('Word Count Distribution', fontsize=14)
        ax4.set_ylabel('Word Count')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        # = os.path.join(viz_dir, dashboard_pathdir)
        plt.savefig(dashboard_pathdir, dpi=300)
        plt.close()

def main():
    """Example usage with multiple PDFs and configurable keywords"""

    # Optional: Define custom keywords for your specific document types
    # Leave as None to use structural analysis only
    custom_header_keywords = None  # e.g., ['PATIENT', 'ROUTINE', 'HEADER']
    custom_data_keywords = None  # e.g., ['RESULTS', 'VALUES', 'DATA']

    # For German medical documents, you could use:
    # custom_header_keywords = ['ROUTINE', 'PATIENT', 'BEFUND']
    # custom_data_keywords = ['ANALYSE', 'WERT', 'NORMWERTE', 'EINHEIT']

    # Initialize analyzer with optional custom keywords
    analyzer = JSONFormatPDFAnalyzer(
        output_dir="json_eda",
        header_keywords=custom_header_keywords,
        data_keywords=custom_data_keywords
    )

    # List of PDFs to process - add your PDF paths here
    pdf_files = [
        "/Users/preethisivakumar/Documents/spacelyout/sample_pdf/Laborbefund.pdf",
        # Add more PDF paths here:
        # "/path/to/another/document.pdf",
        # "/path/to/third/document.pdf",
    ]

    successful_analyses = 0
    failed_analyses = 0

    for pdf_path in pdf_files:
        print(f"\n{'=' * 80}")
        print(f"🔄 PROCESSING: {os.path.basename(pdf_path)}")
        print(f"{'=' * 80}")

        # Check if file exists first
        if not os.path.exists(pdf_path):
            print(f"❌ PDF file not found: {pdf_path}")
            print("Please check the file path and try again.")
            failed_analyses += 1
            continue

        # Run complete analysis with merging
        results = analyzer.run_complete_analysis_with_merging(pdf_path)
        analyzer.create_eda_dashboard(pdf_path,results,"eda")

        if results:
            print(f"\n🎉 Analysis completed successfully for {os.path.basename(pdf_path)}!")
            successful_analyses += 1

            # Show a sample of merged table data
            table_analysis = results['table_analysis']
            if table_analysis:
                print(f"\n📋 Sample from first table:")
                first_table = table_analysis[0]
                print(f"Table {first_table['table_id']}: {first_table['rows']} rows × {first_table['columns']} columns")
                if first_table['rows_data']:
                    print("First few rows:")
                    for i, row in enumerate(first_table['rows_data'][:3]):
                        print(f"  Row {i}: {row}")
                if first_table.get('raw_text'):
                    print(f"Raw text preview: {first_table['raw_text'][:100]}...")
            else:
                print(f"\n📋 No tables were found in the PDF")

            # Show text content preview
            if results and results['content_analysis'].get('full_text'):
                full_text = results['content_analysis']['full_text']
                print(f"\n📄 Text content preview (first 200 chars):")
                print(f"   {full_text[:200]}...")
                print(f"   Total text length: {len(full_text)} characters")
        else:
            print(
                f"\n⚠️  Analysis failed for {os.path.basename(pdf_path)}. Please check the file path and dependencies.")
            failed_analyses += 1

    print(f"\n{'=' * 80}")
    print(f"📊 BATCH PROCESSING SUMMARY")
    print(f"{'=' * 80}")
    print(f"✅ Successful analyses: {successful_analyses}")
    print(f"❌ Failed analyses: {failed_analyses}")
    print(f"📁 All results saved to: json_eda/")

    if successful_analyses > 0:
        print(f"\n💡 TIP: You can now customize table detection by passing:")
        print(f"   - header_keywords=['YOUR', 'HEADER', 'TERMS']")
        print(f"   - data_keywords=['YOUR', 'DATA', 'TERMS']")
        print(f"   to the JSONFormatPDFAnalyzer constructor")


if __name__ == "__main__":
    main()