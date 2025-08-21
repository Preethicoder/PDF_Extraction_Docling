import io
import spacy
from spacy_layout import spaCyLayout
import pandas as pd
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional, Union
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image


class TableMerger:
    def __init__(
            self,
            column_similarity_threshold: float = 0.8,
            page_proximity_limit: int = 2,
            header_text_ratio_threshold: float = 0.7,
            confidence_threshold: float = 0.7,
            enable_fuzzy_matching: bool = False,
            max_merge_chain_length: int = 5,
    ):
        self.column_similarity_threshold = column_similarity_threshold
        self.page_proximity_limit = page_proximity_limit
        self.header_text_ratio_threshold = header_text_ratio_threshold
        self.confidence_threshold = confidence_threshold
        self.enable_fuzzy_matching = enable_fuzzy_matching
        self.max_merge_chain_length = max_merge_chain_length

    def appears_to_have_header(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced header detection with detailed analysis."""
        if df.empty or len(df) < 2:
            return {"has_header": False, "reason": "insufficient_data", "confidence": 0.0}

        first_row = df.iloc[0]

        # Method 1: Check if first row is mostly strings
        text_ratio = sum(1 for val in first_row if isinstance(val, str)) / len(first_row)

        # Method 2: Compare with second row
        if len(df) >= 2:
            second_row = df.iloc[1]
            second_row_numeric = sum(1 for val in second_row
                                     if isinstance(val, (int, float)) or
                                     (isinstance(val, str) and str(val).replace('.', '').replace('-',
                                                                                                 '').isdigit())) / len(
                second_row)
        else:
            second_row_numeric = 0.0

        # Combined confidence score
        confidence = (text_ratio + second_row_numeric) / 2
        has_header = text_ratio > self.header_text_ratio_threshold and second_row_numeric > 0.3

        return {
            "has_header": has_header,
            "text_ratio": text_ratio,
            "second_row_numeric_ratio": second_row_numeric,
            "confidence": confidence,
            "method": "enhanced_heuristic"
        }

    def check_column_similarity(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced column similarity check with fuzzy matching option."""
        details = {
            "threshold": self.column_similarity_threshold,
            "df1_shape": list(df1.shape) if not df1.empty else [0, 0],
            "df2_shape": list(df2.shape) if not df2.empty else [0, 0]
        }

        # Early returns for edge cases
        if df1.empty or df2.empty:
            details.update({"result": False, "reason": "empty_dataframe", "similarity": 0.0})
            return details

        if df1.shape[1] != df2.shape[1]:
            details.update({
                "result": False,
                "reason": "column_count_mismatch",
                "df1_columns": df1.shape[1],
                "df2_columns": df2.shape[1],
                "similarity": 0.0
            })
            return details

        # Column name comparison
        df1_cols = [str(c).strip().lower() for c in df1.columns]
        df2_cols = [str(c).strip().lower() for c in df2.columns]

        # Exact matching
        exact_matches = sum(1 for c1, c2 in zip(df1_cols, df2_cols) if c1 == c2)
        exact_similarity = exact_matches / len(df1_cols) if df1_cols else 0.0

        similarity = exact_similarity
        matching_method = "exact"

        # Fuzzy matching fallback (if enabled and exact matching fails)
        if self.enable_fuzzy_matching and exact_similarity < self.column_similarity_threshold:
            try:
                from difflib import SequenceMatcher
                fuzzy_scores = []
                for c1, c2 in zip(df1_cols, df2_cols):
                    score = SequenceMatcher(None, c1, c2).ratio()
                    fuzzy_scores.append(score)

                fuzzy_similarity = sum(fuzzy_scores) / len(fuzzy_scores)
                if fuzzy_similarity > similarity:
                    similarity = fuzzy_similarity
                    matching_method = "fuzzy"
                    details["fuzzy_scores"] = fuzzy_scores

            except ImportError:
                pass  # Fallback to exact matching if difflib unavailable

        details.update({
            "df1_columns": list(df1.columns),
            "df2_columns": list(df2.columns),
            "exact_matches": exact_matches,
            "total_columns": len(df1_cols),
            "exact_similarity": exact_similarity,
            "final_similarity": similarity,
            "matching_method": matching_method,
            "result": similarity >= self.column_similarity_threshold
        })

        return details

    def evaluate_continuation_criteria(self,
                                       df1: pd.DataFrame,
                                       df2: pd.DataFrame,
                                       metadata1: Dict[str, Any],
                                       metadata2: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive evaluation of continuation table criteria."""

        # Extract metadata with defaults
        heading1 = metadata1.get("heading")
        heading2 = metadata2.get("heading")
        page1 = metadata1.get("page")
        page2 = metadata2.get("page")
        layout1 = metadata1.get("layout", {})
        layout2 = metadata2.get("layout", {})

        criteria = {}

        # 1. Heading similarity
        if heading1 and heading2:
            heading_match = str(heading1).strip() == str(heading2).strip()
            criteria["heading"] = {
                "result": heading_match,
                "heading1": str(heading1).strip(),
                "heading2": str(heading2).strip(),
                "method": "exact_match"
            }
        else:
            heading_match = (heading1 is None and heading2 is None)
            criteria["heading"] = {
                "result": heading_match,
                "heading1": heading1,
                "heading2": heading2,
                "method": "null_match"
            }

        # 2. Page proximity
        if page1 is not None and page2 is not None:
            page_distance = abs(page2 - page1)
            page_proximity = page_distance <= self.page_proximity_limit
            criteria["page_proximity"] = {
                "result": page_proximity,
                "page1": page1,
                "page2": page2,
                "distance": page_distance,
                "limit": self.page_proximity_limit
            }
        else:
            criteria["page_proximity"] = {
                "result": True,  # Assume OK if no page info
                "page1": page1,
                "page2": page2,
                "note": "missing_page_info"
            }
            page_proximity = True

        # 3. Column structure
        column_analysis = self.check_column_similarity(df1, df2)
        criteria["columns"] = column_analysis
        columns_match = column_analysis["result"]

        # 4. Header detection
        header_analysis = self.appears_to_have_header(df2)
        criteria["header_detection"] = header_analysis
        likely_continuation = not header_analysis["has_header"]

        # 5. Bounding box continuity - Fixed attribute access
        bbox_continuity = False

        # Safe attribute access for layout objects
        bbox1 = None
        bbox2 = None

        if hasattr(layout1, 'bbox'):
            bbox1 = layout1.bbox
        elif isinstance(layout1, dict) and 'bbox' in layout1:
            bbox1 = layout1['bbox']

        if hasattr(layout2, 'bbox'):
            bbox2 = layout2.bbox
        elif isinstance(layout2, dict) and 'bbox' in layout2:
            bbox2 = layout2['bbox']

        if bbox1 and bbox2 and page1 is not None and page2 is not None:
            # Check for same page continuation
            if page1 == page2:
                # Table 2 must be immediately below Table 1
                # Need to safely access bbox coordinates
                try:
                    if hasattr(bbox1, 'y2') and hasattr(bbox2, 'y1'):
                        vertically_adjacent = bbox2.y1 > bbox1.y2
                    elif isinstance(bbox1, (list, tuple)) and isinstance(bbox2, (list, tuple)):
                        # Assume format [x1, y1, x2, y2]
                        vertically_adjacent = bbox2[1] > bbox1[3]
                    else:
                        vertically_adjacent = False
                    bbox_continuity = vertically_adjacent
                except (AttributeError, IndexError):
                    bbox_continuity = False
            # Check for across-page continuation
            elif page2 == page1 + 1:
                try:
                    if hasattr(bbox1, 'y2') and hasattr(bbox2, 'y1'):
                        vertically_aligned = bbox1.y2 > 0.85 and bbox2.y1 < 0.15
                    elif isinstance(bbox1, (list, tuple)) and isinstance(bbox2, (list, tuple)):
                        vertically_aligned = bbox1[3] > 0.85 and bbox2[1] < 0.15
                    else:
                        vertically_aligned = False
                    bbox_continuity = vertically_aligned
                except (AttributeError, IndexError):
                    bbox_continuity = False

        criteria["bbox_continuity"] = {
            "result": bbox_continuity,
            "bbox1": str(bbox1) if bbox1 else None,
            "bbox2": str(bbox2) if bbox2 else None,
            "note": "Vertical alignment check across or within pages."
        }

        # Weighted decision scoring
        weights = {
            "heading": 0.2,
            "page_proximity": 0.1,
            "columns": 0.4,
            "no_header": 0.1,
            "bbox_continuity": 0.2
        }

        score = (heading_match * weights["heading"] +
                 page_proximity * weights["page_proximity"] +
                 columns_match * weights["columns"] +
                 likely_continuation * weights["no_header"] +
                 bbox_continuity * weights["bbox_continuity"])

        # Both strict AND confidence-based decisions
        strict_continuation = all([heading_match, page_proximity, columns_match, likely_continuation, bbox_continuity])
        confidence_continuation = score >= self.confidence_threshold

        criteria["final_decision"] = {
            "is_continuation": strict_continuation,
            "confidence_score": score,
            "confidence_continuation": confidence_continuation,
            "passed_criteria": {
                "heading": heading_match,
                "page_proximity": page_proximity,
                "columns": columns_match,
                "no_header": likely_continuation,
                "bbox_continuity": bbox_continuity
            },
            "weights_used": weights
        }

        return criteria

    def merge_table_group(self, table_group: List[Dict[str, Any]]) -> pd.DataFrame:
        """Merge a group of table dictionaries into a single DataFrame."""
        if not table_group:
            return pd.DataFrame()

        dataframes = []
        for table_dict in table_group:
            df = table_dict.get("data", pd.DataFrame()).copy()
            if not df.empty:
                # Sanitize columns before adding to the merge list
                df = self._sanitize_columns(df)
                dataframes.append(df)

        if not dataframes:
            return pd.DataFrame()

        try:
            # pd.concat with ignore_index=True is the correct way for row-wise merges
            merged_df = pd.concat(dataframes, ignore_index=True)
            return self.clean_merged_table(merged_df)
        except Exception as e:
            print(f"Warning: Error merging tables: {e}")
            return dataframes[0] if dataframes else pd.DataFrame()

    def clean_merged_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate headers that may appear in the middle of merged tables."""
        if df.empty or len(df) < 3:
            return df

        header_row = df.iloc[0]
        rows_to_remove = []

        for i in range(1, len(df)):
            current_row = df.iloc[i]

            # Check similarity to header row
            similarity = sum(1 for h, c in zip(header_row, current_row)
                             if str(h).strip().lower() == str(c).strip().lower())

            if len(header_row) > 0 and similarity >= len(header_row) * 0.8:
                rows_to_remove.append(i)

        if rows_to_remove:
            # Use set to ensure unique indices are dropped
            unique_rows_to_remove = sorted(list(set(rows_to_remove)))
            df = df.drop(df.index[unique_rows_to_remove]).reset_index(drop=True)

        return df

    def _sanitize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures DataFrame columns are unique."""
        if df.columns.is_unique:
            return df

        # Create a list of new, unique column names
        new_cols = []
        seen = {}
        for col in df.columns:
            if col in seen:
                seen[col] += 1
                new_cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                new_cols.append(col)

        df.columns = new_cols
        return df

    def merge_tables(self, tables: List[Dict[str, Any]]) -> Tuple[List[pd.DataFrame], List[Dict[str, Any]]]:
        """
        Main method to merge continuation tables with chain length protection.

        Args:
            tables: List of table dictionaries with keys: 'data', 'heading', 'page', etc.

        Returns:
            Tuple of (merged_tables, merge_details)
        """
        merged_tables = []
        merge_details = []
        processed_indices = set()

        for i, table in enumerate(tables):
            if i in processed_indices:
                continue

            # Start new merge group
            table_group = [table]
            current_metadata = {
                "heading": table.get("heading"),
                "page": table.get("page", 1),
                "layout": table.get("layout", {})
            }

            # Track merge details
            group_details = {
                "merge_group_id": len(merged_tables) + 1,
                "primary_table_index": i,
                "tables_in_group": [self._table_summary(i, table)],
                "merge_decisions": []
            }

            # Look for continuation tables with chain length protection
            chain_length = 1
            for j in range(i + 1, len(tables)):
                if j in processed_indices or chain_length >= self.max_merge_chain_length:
                    if chain_length >= self.max_merge_chain_length:
                        group_details["merge_decisions"].append({
                            "note": f"Stopped merging at max chain length: {self.max_merge_chain_length}"
                        })
                    break

                candidate = tables[j]
                candidate_metadata = {
                    "heading": candidate.get("heading"),
                    "page": candidate.get("page", 1),
                    "layout": candidate.get("layout", {})
                }

                # Evaluate if this is a continuation
                criteria = self.evaluate_continuation_criteria(
                    table.get("data", pd.DataFrame()),
                    candidate.get("data", pd.DataFrame()),
                    current_metadata,
                    candidate_metadata
                )

                # Use either strict or confidence-based decision
                is_continuation = (criteria["final_decision"]["is_continuation"] or
                                   criteria["final_decision"].get("confidence_continuation", False))

                decision_record = {
                    "candidate_index": j,
                    "decision": "accepted" if is_continuation else "rejected",
                    "criteria": criteria,
                    "chain_position": chain_length + 1
                }

                if is_continuation:
                    table_group.append(candidate)
                    processed_indices.add(j)
                    # Update metadata for next comparison
                    current_metadata = candidate_metadata
                    group_details["tables_in_group"].append(self._table_summary(j, candidate))
                    chain_length += 1

                group_details["merge_decisions"].append(decision_record)

            # Merge the group and finalize details
            merged_df = self.merge_table_group(table_group)
            group_details.update({
                "tables_merged_count": len(table_group),
                "merge_type": "merged_tables" if len(table_group) > 1 else "single_table",
                "final_shape": list(merged_df.shape) if not merged_df.empty else [0, 0],
                "final_columns": list(merged_df.columns) if not merged_df.empty else [],
                "chain_length": chain_length
            })

            merged_tables.append(merged_df)
            merge_details.append(group_details)
            processed_indices.add(i)

        return merged_tables, merge_details

    def _table_summary(self, index: int, table: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of a table for merge details."""
        df = table.get("data", pd.DataFrame())
        return {
            "original_index": index,
            "page": table.get("page"),
            "heading": table.get("heading"),
            "shape": list(df.shape) if not df.empty else [0, 0],
            "columns": list(df.columns) if not df.empty else []
        }


class TableProcessor:
    """High-level processor that handles document processing and file I/O."""

    def __init__(self, merger: Optional[TableMerger] = None):
        self.merger = merger or TableMerger()

    def process_spacy_document(self, document_path: str) -> Tuple[List[pd.DataFrame], List[Dict[str, Any]]]:
        """
        Process a document using spaCy-layout.

        Saves as text and a single image in all cases.
        """
        try:
            nlp = spacy.blank("en")
            layout = spaCyLayout(nlp)

            doc = layout(document_path)
            tables = []

            # Save the document text and images regardless of whether tables are found
            document_name = Path(document_path).stem
            self.save_as_text(doc.text, document_name)
            self.save_images(document_path, document_name)

            # Extract tables from the document
            for table_span in doc._.tables:
                table_dict = {
                    "data": table_span._.data,
                    "heading": table_span._.heading.text if table_span._.heading else None,
                    "page": table_span._.layout.page_no if table_span._.layout else None,
                    "layout": table_span._.layout if table_span._.layout else None
                }
                tables.append(table_dict)

            # If no tables were found, return empty lists.
            if not tables:
                print("No tables found in document.")
                return [], []

            # If tables were found, proceed with merging logic
            return self.merger.merge_tables(tables)

        except Exception as e:
            print(f"Error processing document: {e}")
            return [], []

    def process_mock_tables(self, tables: List[Dict[str, Any]]) -> Tuple[List[pd.DataFrame], List[Dict[str, Any]]]:
        """Process mock tables for testing."""
        return self.merger.merge_tables(tables)

    def save_as_text(self, text_content: str, document_name: str, output_dir: str = "output") -> str:
        """Saves the document content as a plain text file."""
        try:
            Path(output_dir).mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            txt_path = os.path.join(output_dir, f"{document_name}_{timestamp}.txt")

            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text_content)

            print(f"Saved document content to: {txt_path}")
            return txt_path
        except Exception as e:
            print(f"Error saving text file: {e}")
            return ""

    def save_images(self, document_path: str, document_name: str, output_dir: str = "output") -> Optional[str]:
        """Saves all embedded images from the PDF as a single combined PNG file."""
        try:
            doc = fitz.open(document_path)
            embedded_images = []

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images(full=True)

                for img_info in image_list:
                    xref = img_info[0]
                    try:
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        img = Image.open(io.BytesIO(image_bytes))
                        embedded_images.append(img)
                    except Exception as e:
                        print(f"Warning: Could not process image with xref {xref}: {e}")
                        continue

            if not embedded_images:
                print("No embedded images found in the document.")
                return None

            # Determine the dimensions for the combined image
            total_height = sum(img.height for img in embedded_images)
            max_width = max(img.width for img in embedded_images)

            # Create the new canvas
            combined_image = Image.new("RGB", (max_width, total_height), "white")

            # Paste images onto the canvas
            current_y = 0
            for img in embedded_images:
                combined_image.paste(img, (0, current_y))
                current_y += img.height

            Path(output_dir).mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(output_dir, f"{document_name}_embedded_images_{timestamp}.png")
            combined_image.save(image_path)

            print(f"Saved all embedded images as a single file to: {image_path}")
            return image_path

        except Exception as e:
            print(f"Error saving images: {e}")
            return None

    def save_results(self,
                     merged_tables: List[pd.DataFrame],
                     merge_details: List[Dict[str, Any]],
                     document_name: str,
                     output_dir: str = "output") -> Dict[str, Any]:
        """Save results with enhanced metadata."""
        try:
            Path(output_dir).mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save combined CSV
            csv_path = None
            if merged_tables:
                combined_data = []
                for i, table in enumerate(merged_tables):
                    if not table.empty:
                        table_copy = table.copy()
                        table_copy['_table_source'] = f"Table_{i + 1}"
                        table_copy['_merge_group'] = i + 1
                        combined_data.append(table_copy)

                if combined_data:
                    combined_df = pd.concat(combined_data, ignore_index=True, sort=False)
                    csv_path = os.path.join(output_dir, f"{document_name}_merged_{timestamp}.csv")
                    combined_df.to_csv(csv_path, index=False, encoding='utf-8')
                    print(f"Saved merged tables to: {csv_path}")

            # Save detailed JSON
            json_data = {
                "processing_info": {
                    "timestamp": datetime.now().isoformat(),
                    "document_name": document_name,
                    "merger_config": {
                        "column_similarity_threshold": self.merger.column_similarity_threshold,
                        "page_proximity_limit": self.merger.page_proximity_limit,
                        "header_text_ratio_threshold": self.merger.header_text_ratio_threshold,
                        "confidence_threshold": self.merger.confidence_threshold,
                        "enable_fuzzy_matching": self.merger.enable_fuzzy_matching,
                        "max_merge_chain_length": self.merger.max_merge_chain_length
                    }
                },
                "results_summary": {
                    "total_merged_tables": len(merged_tables),
                    "total_original_tables": sum(
                        len(detail["tables_in_group"]) for detail in merge_details) if merge_details else 0,
                    "successfully_merged_groups": sum(
                        1 for detail in merge_details if
                        detail["merge_type"] == "merged_tables") if merge_details else 0
                },
                "merge_details": merge_details
            }

            json_path = os.path.join(output_dir, f"{document_name}_details_{timestamp}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

            print(f"Saved merge details to: {json_path}")

            return {"csv_path": csv_path, "json_path": json_path}

        except Exception as e:
            print(f"Error saving results: {e}")
            return {"csv_path": None, "json_path": None}


# Example usage
if __name__ == "__main__":
    # Example 1: Mock data testing
    """print("=== Testing with Mock Data ===")

    mock_tables = [
        {"data": pd.DataFrame({"Name": ["Alice", "Bob"], "Age": [25, 30]}), "heading": "People", "page": 1},
        {"data": pd.DataFrame({"Name": ["Charlie", "Diana"], "Age": [35, 28]}), "heading": "People", "page": 2},
        {"data": pd.DataFrame({"Product": ["Laptop", "Mouse"], "Price": [1000, 25]}), "heading": "Items", "page": 3}
    ]

    processor = TableProcessor()
    merged_tables, details = processor.process_mock_tables(mock_tables)

    print(f"Found {len(merged_tables)} merged table groups:")
    for i, table in enumerate(merged_tables):
        print(f"  Group {i + 1}: {table.shape}")
        if i < len(details):
            print(f"    Merge type: {details[i]['merge_type']}")
            print(f"    Original tables: {details[i]['tables_merged_count']}")"""

    # Example 2: Real document processing#
    processor = TableProcessor()
    print("\n=== Processing Real Document ===")
    document_path = "/Users/preethisivakumar/Documents/spacelyout/sample_pdf/Laborbefund.pdf"  # Update this path

    if os.path.exists(document_path):
        try:
            merged_tables, details = processor.process_spacy_document(document_path)
            if merged_tables:
                results = processor.save_results(merged_tables, details, Path(document_path).stem)
                print(f"Processing completed. Results: {results}")
            else:
                print("No tables found or processed.")
        except Exception as e:
            print(f"Error processing document: {e}")
    else:
        print(f"Document not found: {document_path}")
        print("Please update the document_path variable with a valid PDF file path.")