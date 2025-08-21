#!/usr/bin/env python3
"""
PDF Analyzer with JSON Output Format and Table Merging

Creates JSON output in your specified format with table merging and complete data extraction.

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
            header_confidence_threshold: float = 0.7
    ):
        self.column_similarity_threshold = column_similarity_threshold
        self.page_proximity_limit = page_proximity_limit
        self.header_confidence_threshold = header_confidence_threshold

    def is_header_table(self, table_info: Dict[str, Any]) -> bool:
        """Detect header tables (like ROUTINE tables)"""
        if table_info.get('rows', 0) != 1:
            return False

        column_names = table_info.get('column_names', [])
        if not column_names:
            return False

        # Check for typical header keywords
        text_content = ' '.join(str(col) for col in column_names).upper()
        header_keywords = ['ROUTINE', 'SEITE', 'GEB:', 'ZUWEISUNG', 'BEFUNDDATUM', 'MUSTERMANN']

        return any(keyword in text_content for keyword in header_keywords)

    def is_data_table(self, table_info: Dict[str, Any]) -> bool:
        """Detect data tables (like analysis tables)"""
        if table_info.get('rows', 0) <= 1:
            return False

        column_names = table_info.get('column_names', [])
        if not column_names:
            return False

        # Check for typical data table keywords
        text_content = ' '.join(str(col) for col in column_names).lower()
        data_keywords = ['analyse', 'einheit', 'normwerte', 'wert', 'result', 'value']

        return any(keyword in text_content for keyword in data_keywords)

    def calculate_column_similarity(self, cols1: List[str], cols2: List[str]) -> float:
        """Calculate similarity between two column sets"""
        if not cols1 or not cols2 or len(cols1) != len(cols2):
            return 0.0

        # Clean and normalize column names
        clean_cols1 = [str(col).strip().lower() for col in cols1]
        clean_cols2 = [str(col).strip().lower() for col in cols2]

        # Calculate exact matches
        exact_matches = sum(1 for c1, c2 in zip(clean_cols1, clean_cols2) if c1 == c2)
        exact_similarity = exact_matches / len(cols1)

        # If not perfect match, try fuzzy matching
        if exact_similarity < 1.0:
            fuzzy_scores = []
            for c1, c2 in zip(clean_cols1, clean_cols2):
                score = SequenceMatcher(None, c1, c2).ratio()
                fuzzy_scores.append(score)
            fuzzy_similarity = sum(fuzzy_scores) / len(fuzzy_scores)
            return max(exact_similarity, fuzzy_similarity)

        return exact_similarity

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

        filtered_rows = []
        ref_cols_lower = [str(col).strip().lower() for col in reference_columns]

        for row in rows_data:
            row_lower = [str(val).strip().lower() for val in row]

            # Check similarity to reference columns
            similarity_count = sum(1 for val, ref in zip(row_lower, ref_cols_lower) if val == ref)
            similarity_ratio = similarity_count / len(reference_columns) if reference_columns else 0

            # Keep row if it's not too similar to header
            if similarity_ratio < 0.7:
                filtered_rows.append(row)

        return filtered_rows

    def process_tables(self, tables_analysis: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Process and merge tables while maintaining JSON format"""
        if not tables_analysis:
            return [], {}

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

        # Group data tables for merging
        merged_tables = []
        merge_info = {
            'original_table_count': len(tables_analysis),
            'header_tables': len(header_tables),
            'data_tables': len(data_tables),
            'other_tables': len(other_tables),
            'merge_groups': []
        }

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

    def __init__(self, output_dir="json_eda"):
        """Initialize the analyzer"""
        print("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm")
        self.layout_processor = spaCyLayout(self.nlp)
        self.doc = None
        self.output_dir = output_dir
        self.table_merger = SmartTableMerger()

        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"✓ Created output directory: {self.output_dir}")

    def load_pdf(self, pdf_path):
        """Load and process PDF"""
        print(f"Processing: {pdf_path}")
        self.doc = self.layout_processor(pdf_path)
        self.doc = self.nlp(self.doc)
        print("✓ PDF loaded successfully!")

    def basic_stats(self):
        """Get basic document statistics"""
        if not self.doc:
            return {}

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

    def analyze_layout(self):
        """Analyze document layout structure"""
        if not self.doc or 'layout' not in self.doc.spans:
            return {}

        layout_spans = self.doc.spans['layout']
        span_types = Counter(span.label_ for span in layout_spans)

        return span_types

    def analyze_text_content(self):
        """Analyze text content"""
        if not self.doc:
            return {}

        total_tokens = len([t for t in self.doc if not t.is_space])

        return {
            'total_tokens': total_tokens
        }

    def extract_table_data(self, table_obj) -> List[List[Any]]:
        """Extract all row data from a table object"""
        if not hasattr(table_obj._, 'data') or table_obj._.data is None:
            return []

        df = table_obj._.data
        if df.empty:
            return []

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

        return rows_data

    def analyze_tables_with_data(self):
        """Analyze tables and extract complete data in your JSON format"""
        if not self.doc or not hasattr(self.doc._, 'tables'):
            return []

        tables = self.doc._.tables
        table_analysis = []

        for i, table in enumerate(tables):
            # Extract table data
            rows_data = self.extract_table_data(table)

            table_info = {
                'table_id': i + 1,
                'position': f"Tokens {table.start}-{table.end}",
                'rows': len(rows_data),
                'columns': len(rows_data[0]) if rows_data else 0,
                'column_names': rows_data[0] if rows_data else [],
                'rows_data': rows_data
            }

            # Add page information if available
            if hasattr(table._, 'layout') and hasattr(table._.layout, 'page'):
                table_info['page'] = table._.layout.page
            else:
                table_info['page'] = 1

            table_analysis.append(table_info)

        return table_analysis

    def run_complete_analysis_with_merging(self, pdf_path):
        """Run complete analysis with table merging in your JSON format"""
        print("🔍 Starting JSON Format PDF Analysis with Table Merging")
        print("=" * 60)

        try:
            # Load PDF
            self.load_pdf(pdf_path)

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
            print(f"🔄 Merge groups created: {len(merge_info['merge_groups'])}")

            # Create final results in your format
            results = {
                'basic_stats': basic_stats,
                'layout_analysis': dict(layout_analysis),
                'content_analysis': content_analysis,
                'table_analysis': processed_tables,
                'merge_info': merge_info
            }

            # Save complete results
            results_path = os.path.join(self.output_dir, "complete_analysis_with_merging.json")
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

            table_results_path = os.path.join(self.output_dir, "table_analysis_merged.json")
            with open(table_results_path, 'w', encoding='utf-8') as f:
                json.dump(table_only_results, f, indent=2, ensure_ascii=False, default=str)

            print(f"✅ Table analysis (your format) saved to: {table_results_path}")

            # Print summary
            self.print_analysis_summary(results)

            return results

        except Exception as e:
            print(f"❌ Error during analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def print_analysis_summary(self, results):
        """Print a summary of the analysis"""
        print(f"\n📊 ANALYSIS SUMMARY")
        print("=" * 40)

        basic_stats = results['basic_stats']
        print(f"Pages: {basic_stats.get('Number of Pages', 'Unknown')}")
        print(f"Total Tokens: {basic_stats.get('Total Tokens', 0)}")
        print(f"Total Tables: {len(results['table_analysis'])}")

        merge_info = results['merge_info']
        print(f"Original Tables: {merge_info['original_table_count']}")
        print(f"Header Tables: {merge_info['header_tables']}")
        print(f"Data Tables: {merge_info['data_tables']}")
        print(f"Merge Groups: {len(merge_info['merge_groups'])}")

        print(f"\n📋 TABLE DETAILS:")
        for table in results['table_analysis']:
            table_type = "HEADER" if len(table.get('merged_from', [])) == 0 and table['rows'] == 1 else "DATA"
            merged_info = f" (merged from tables {table.get('merged_from', [])})" if 'merged_from' in table else ""

            print(
                f"  Table {table['table_id']}: {table['rows']} rows × {table['columns']} columns [{table_type}]{merged_info}")
            if table['column_names']:
                print(
                    f"    Columns: {', '.join(table['column_names'][:3])}{'...' if len(table['column_names']) > 3 else ''}")

        print(f"\n📁 Files generated:")
        print(f"  - complete_analysis_with_merging.json (with merge info)")
        print(f"  - table_analysis_merged.json (your exact format)")


def main():
    """Example usage"""
    # Initialize analyzer
    analyzer = JSONFormatPDFAnalyzer(output_dir="json_eda")

    # Replace with your PDF path
    pdf_path = "/Users/preethisivakumar/Documents/spacelyout/sample_pdf/arztbrief_innere_medizin.pdf"

    # Run complete analysis with merging
    results = analyzer.run_complete_analysis_with_merging(pdf_path)

    if results:
        print(f"\n🎉 Analysis completed successfully!")
        print(f"📁 Check the 'json_eda' folder for results")

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
    else:
        print("\n⚠️  Analysis failed. Please check the file path and dependencies.")


if __name__ == "__main__":
    main()