import spacy
from spacy_layout import spaCyLayout
import pandas as pd
from typing import List, Dict, Tuple, Any

# ------------------------
# Utility functions
# ------------------------
def appears_to_have_header(df: pd.DataFrame) -> bool:
    """Simple check: first row might be header if values are strings."""
    if df.empty:
        return False
    first_row = df.iloc[0]
    return all(isinstance(x, str) for x in first_row)

def check_column_similarity_detailed(df1: pd.DataFrame, df2: pd.DataFrame, threshold: float = 0.8) -> Tuple[bool, Dict[str, Any]]:
    details = {"threshold": threshold, "df1_shape": list(df1.shape), "df2_shape": list(df2.shape)}
    if df1.empty or df2.empty:
        details["result"] = False
        details["reason"] = "One or both dataframes are empty"
        return False, details
    if df1.shape[1] != df2.shape[1]:
        details["result"] = False
        details["reason"] = "Different number of columns"
        return False, details
    df1_cols = [str(c).strip().lower() for c in df1.columns]
    df2_cols = [str(c).strip().lower() for c in df2.columns]
    matching_cols = sum(1 for c1, c2 in zip(df1_cols, df2_cols) if c1 == c2)
    similarity = matching_cols / len(df1_cols)
    details["result"] = similarity >= threshold
    details["similarity"] = similarity
    return similarity >= threshold, details

def is_continuation_table_detailed(df1: pd.DataFrame, df2: pd.DataFrame, heading1, heading2, page1: int, page2: int) -> Tuple[bool, Dict[str, Any]]:
    criteria_results = {}

    # Heading check
    if heading1 and heading2:
        same_heading = heading1.strip() == heading2.strip()
    else:
        same_heading = heading1 is None and heading2 is None
    criteria_results["heading_match"] = same_heading

    # Page proximity
    page_proximity = abs(page2 - page1) <= 2 if page1 is not None and page2 is not None else True
    criteria_results["page_proximity"] = page_proximity

    # Column similarity
    columns_match, col_details = check_column_similarity_detailed(df1, df2)
    criteria_results["column_similarity"] = col_details

    # Header detection
    likely_continuation = not appears_to_have_header(df2)
    criteria_results["header_detection"] = likely_continuation

    is_continuation = same_heading and page_proximity and columns_match and likely_continuation
    criteria_results["final_decision"] = is_continuation

    return is_continuation, criteria_results

def merge_table_group(tables: List[Any]) -> pd.DataFrame:
    dfs = [t["_data"] for t in tables]
    merged = pd.concat(dfs, ignore_index=True)
    return merged

def merge_continuation_tables(tables: List[Dict[str, Any]]) -> Tuple[List[pd.DataFrame], List[Dict[str, Any]]]:
    merged_tables = []
    merge_details = []
    processed_indices = set()

    for i, table in enumerate(tables):
        if i in processed_indices:
            continue

        table_group = [table]
        current_df = table["_data"]
        current_heading = table.get("_heading")
        current_page = table.get("_page", 1)

        group_details = {
            "merge_group_id": len(merged_tables) + 1,
            "tables_in_group": [table],
            "merge_decisions": []
        }

        for j in range(i + 1, len(tables)):
            if j in processed_indices:
                continue
            candidate = tables[j]
            candidate_df = candidate["_data"]
            candidate_heading = candidate.get("_heading")
            candidate_page = candidate.get("_page", 1)

            is_continuation, criteria = is_continuation_table_detailed(
                current_df, candidate_df, current_heading, candidate_heading, current_page, candidate_page
            )

            decision = "accepted" if is_continuation else "rejected"
            group_details["merge_decisions"].append({"candidate_index": j, "decision": decision, "criteria": criteria})

            if is_continuation:
                table_group.append(candidate)
                processed_indices.add(j)
                current_page = candidate_page

        merged_df = merge_table_group(table_group)
        merged_tables.append(merged_df)
        merge_details.append(group_details)
        processed_indices.add(i)

    return merged_tables, merge_details

# ------------------------
# Example usage with mock data
# ------------------------
if __name__ == "__main__":
    # Mock tables (simulate tables extracted via spaCy-layout)
    table1 = {"_data": pd.DataFrame({"A": [1, 2], "B": [3, 4]}), "_heading": "Table 1", "_page": 1}
    table2 = {"_data": pd.DataFrame({"A": [5, 6], "B": [7, 8]}), "_heading": "Table 1", "_page": 2}
    table3 = {"_data": pd.DataFrame({"X": [9, 10], "Y": [11, 12]}), "_heading": "Table 2", "_page": 2}

    tables = [table1, table2, table3]

    merged_tables, merge_info = merge_continuation_tables(tables)

    for idx, df in enumerate(merged_tables):
        print(f"\nMerged Table {idx + 1}:\n", df)
        print("Merge Info:", merge_info[idx])
