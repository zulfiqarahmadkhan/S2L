"""Generate master summary sheets from ROI Excel files."""
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def generate_summary_sheet(directory: str, save_path: str, files: list | None = None):
    """Generate a summary Excel sheet from ROI Excel files.

    *files* limits the summary to those sheets (e.g. the ones written by the
    current run); otherwise every Excel file in *directory* is used.
    """
    directory_path = Path(directory)
    if files is not None:
        excel_files = [Path(f) for f in files if Path(f).exists()]
    else:
        excel_files = list(directory_path.glob("*.xlsx")) + list(directory_path.glob("*.xls"))

    if not excel_files:
        logger.warning(f"No Excel files found in: {directory}")
        return

    import numpy as np

    def parse_coords(series):
        xs, ys = [], []
        for val in series.dropna().astype(str):
            val = val.strip("() ")
            if "," in val:
                parts = val.split(",")
                try:
                    xs.append(float(parts[0]))
                    ys.append(float(parts[1]))
                except ValueError:
                    pass
        if not xs:
            return ""
        return f"({round(np.mean(xs), 2)}, {round(np.mean(ys), 2)})"

    def parse_min_max(series):
        mins, maxs = [], []
        for val in series.dropna().astype(str):
            if "-" in val:
                parts = val.split("-")
                try:
                    mins.append(float(parts[0]))
                    maxs.append(float(parts[1]))
                except ValueError:
                    pass
        if not mins:
            return ""
        return f"{int(round(np.mean(mins)))}-{int(round(np.mean(maxs)))}"

    def parse_bbox(series):
        xs, ys, ws, hs = [], [], [], []
        for val in series.dropna().astype(str):
            try:
                parts = val.split(",")
                d = {}
                for p in parts:
                    k, v = p.split("=")
                    d[k.strip()] = float(v.strip())
                xs.append(d["x"])
                ys.append(d["y"])
                ws.append(d["w"])
                hs.append(d["h"])
            except Exception:
                pass
        if not xs:
            return ""
        return f"x={round(np.mean(xs), 1)}, y={round(np.mean(ys), 1)}, w={round(np.mean(ws), 1)}, h={round(np.mean(hs), 1)}"

    def parse_ellipse(series):
        majors, minors, angles = [], [], []
        for val in series.dropna().astype(str):
            try:
                parts = val.split(",")
                d = {}
                for p in parts:
                    k, v = p.split("=")
                    d[k.strip()] = float(v.strip())
                majors.append(d["major"])
                minors.append(d["minor"])
                angles.append(d["angle"])
            except Exception:
                pass
        if not majors:
            return ""
        return f"major={round(np.mean(majors), 2)}, minor={round(np.mean(minors), 2)}, angle={round(np.mean(angles), 1)}"

    def parse_shape(series):
        circs, solds = [], []
        for val in series.dropna().astype(str):
            try:
                parts = val.split(",")
                d = {}
                for p in parts:
                    k, v = p.split("=")
                    d[k.strip()] = float(v.strip())
                circs.append(d["circularity"])
                solds.append(d["solidity"])
            except Exception:
                pass
        if not circs:
            return ""
        return f"circularity={round(np.mean(circs), 3)}, solidity={round(np.mean(solds), 3)}"

    results = []
    for file_path in excel_files:
        if file_path.stem == "Summary":
            continue
        try:
            logger.info(f"Processing: {file_path}")
            df = pd.read_excel(file_path)

            # an image with 0 cells has a sheet with only the header row: keep it (0 objects)
            if 'Label' not in df.columns or 'Integrated Density' not in df.columns:
                logger.warning(f"Skipping file with missing columns: {file_path}")
                continue

            # Base dictionary with the original 3 attributes
            row_data = {
                'File Location': str(file_path),
                'Number of Objects': int(df['Label'].dropna().shape[0]),
                'Sum of Integrated Density': float(df['Integrated Density'].sum()),
            }

            # Add Mean of all defined columns
            # Column mapping & custom parsing for non-numeric columns
            if 'Area' in df.columns:
                row_data['Mean Area'] = round(float(df['Area'].mean()), 2)
            if 'Integrated Density' in df.columns:
                row_data['Mean Integrated Density'] = round(float(df['Integrated Density'].mean()), 2)
            if 'Mean Gray Value' in df.columns:
                row_data['Mean Mean Gray Value'] = round(float(df['Mean Gray Value'].mean()), 4)
            if 'Standard Deviation' in df.columns:
                row_data['Mean Standard Deviation'] = round(float(df['Standard Deviation'].mean()), 4)
            if 'Modal gray value' in df.columns:
                row_data['Mean Modal gray value'] = round(float(df['Modal gray value'].mean()), 2)
            if 'Centroid' in df.columns:
                row_data['Mean Centroid'] = parse_coords(df['Centroid'])
            if 'Perimeter' in df.columns:
                row_data['Mean Perimeter'] = round(float(df['Perimeter'].mean()), 4)
            if 'Min & max gray value' in df.columns:
                row_data['Mean Min & max gray value'] = parse_min_max(df['Min & max gray value'])
            if 'Center of mass' in df.columns:
                row_data['Mean Center of mass'] = parse_coords(df['Center of mass'])
            if 'Bounding rectangle' in df.columns:
                row_data['Mean Bounding rectangle'] = parse_bbox(df['Bounding rectangle'])
            if 'Fit ellipse' in df.columns:
                row_data['Mean Fit ellipse'] = parse_ellipse(df['Fit ellipse'])
            if 'Shape descriptors' in df.columns:
                row_data['Mean Shape descriptors'] = parse_shape(df['Shape descriptors'])
            if "Feret's diameter" in df.columns:
                row_data["Mean Feret's diameter"] = round(float(df["Feret's diameter"].mean()), 4)
            if 'Skewness' in df.columns:
                row_data['Mean Skewness'] = round(float(df['Skewness'].mean()), 4)
            if 'Area fraction' in df.columns:
                row_data['Mean Area fraction'] = round(float(df['Area fraction'].mean()), 4)
            if 'Median' in df.columns:
                row_data['Mean Median'] = round(float(df['Median'].mean()), 4)
            if 'Kurtosis' in df.columns:
                row_data['Mean Kurtosis'] = round(float(df['Kurtosis'].mean()), 4)
            if 'Stack position' in df.columns:
                row_data['Mean Stack position'] = round(float(df['Stack position'].mean()), 2)

            results.append(row_data)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    if not results:
        logger.warning("No valid Excel files were processed.")
        return

    results_df = pd.DataFrame(results)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
        results_df.to_excel(writer, index=False, sheet_name='Summary')
        workbook = writer.book
        worksheet = writer.sheets['Summary']

        header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
        for col_num, value in enumerate(results_df.columns):
            worksheet.write(0, col_num, value, header_fmt)

        for i, col in enumerate(results_df.columns):
            max_len = max(results_df[col].astype(str).map(len).max(), len(str(col))) + 2
            worksheet.set_column(i, i, max_len)

    logger.info(f"Summary generated: {save_path}")
