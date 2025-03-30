from flask import Blueprint, render_template, request, session, redirect, url_for, flash, jsonify, current_app
import pandas as pd
import os
import sys
import zipfile
import io
from datetime import datetime

# Add the parent directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_processing.data_loader import get_latest_data, consolidate_csv_files, update_data, get_data_summary
from src.data_processing.data_loader import load_from_csv, merge_dataframes, save_to_csv

bp = Blueprint('data', __name__, url_prefix='/data')

@bp.route('/')
def index():
    """Data Management page"""
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    # Get data summary
    data_summary = get_data_summary(data_dir=data_dir)
    session['data_summary'] = data_summary
    
    return render_template('data.html', 
                           data_summary=data_summary, 
                           active_tab='overview')

@bp.route('/refresh', methods=['POST'])
def refresh():
    """Refresh data (from sidebar)"""
    # Clear session data to force reload
    if 'trader' in session:
        session.pop('trader')
    if 'nifty_data' in session:
        session.pop('nifty_data')
    if 'vix_data' in session:
        session.pop('vix_data')
    
    # Get updated data summary
    data_dir = 'data'
    data_summary = get_data_summary(data_dir=data_dir)
    session['data_summary'] = data_summary
    
    flash("Data refreshed successfully", "success")
    return redirect(url_for('main.index'))

@bp.route('/update', methods=['POST'])
def update():
    """Update data with new fetches"""
    try:
        days = int(request.form.get('days', 30))
        
        # Fetch and merge data
        nifty_df, vix_df = update_data(days=days, data_dir='data')
        
        # Update session data summary
        data_summary = get_data_summary(data_dir='data')
        session['data_summary'] = data_summary
        
        flash(f"Successfully updated data with {len(nifty_df)} NIFTY records and {len(vix_df)} VIX records", "success")
    except Exception as e:
        flash(f"Error updating data: {e}", "error")
    
    return redirect(url_for('data.index'))

@bp.route('/consolidate', methods=['POST'])
def consolidate():
    """Consolidate multiple data files into one"""
    try:
        # Consolidate files
        nifty_df, vix_df = consolidate_csv_files(data_dir='data')
        
        # Update session data summary
        data_summary = get_data_summary(data_dir='data')
        session['data_summary'] = data_summary
        
        flash(f"Successfully consolidated data with {len(nifty_df)} NIFTY records and {len(vix_df)} VIX records", "success")
    except Exception as e:
        flash(f"Error consolidating files: {e}", "error")
    
    return redirect(url_for('data.index'))

@bp.route('/delete-old', methods=['POST'])
def delete_old():
    """Delete old files after consolidation"""
    try:
        data_dir = 'data'
        # Get data summary to find files
        data_summary = get_data_summary(data_dir=data_dir)
        
        # Only delete non-consolidated files
        count = 0
        for file_info in data_summary.get('other_files', []):
            if file_info['name'] not in ['nifty_data_consolidated.csv', 'vix_data_consolidated.csv']:
                file_path = os.path.join(data_dir, file_info['name'])
                os.remove(file_path)
                count += 1
        
        # Update session data summary
        data_summary = get_data_summary(data_dir=data_dir)
        session['data_summary'] = data_summary
        
        flash(f"Successfully deleted {count} old files", "success")
    except Exception as e:
        flash(f"Error deleting files: {e}", "error")
    
    return redirect(url_for('data.index'))

@bp.route('/export', methods=['POST'])
def export_data():
    """Export consolidated data"""
    try:
        export_format = request.form.get('export_format', 'CSV')
        data_dir = 'data'
        
        # Load consolidated data
        nifty_consolidated_path = os.path.join(data_dir, 'nifty_data_consolidated.csv')
        vix_consolidated_path = os.path.join(data_dir, 'vix_data_consolidated.csv')
        
        if not os.path.exists(nifty_consolidated_path) or not os.path.exists(vix_consolidated_path):
            flash("No consolidated data files found. Please consolidate your data first.", "warning")
            return redirect(url_for('data.index'))
        
        nifty_df, vix_df = load_from_csv(nifty_consolidated_path, vix_consolidated_path)
        
        # Create a zip file with both datasets
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED) as zip_file:
            if export_format == "CSV":
                # Write CSVs to zip
                nifty_buffer = io.StringIO()
                vix_buffer = io.StringIO()
                
                nifty_df.to_csv(nifty_buffer)
                vix_df.to_csv(vix_buffer)
                
                zip_file.writestr("nifty_data.csv", nifty_buffer.getvalue())
                zip_file.writestr("vix_data.csv", vix_buffer.getvalue())
            
            elif export_format == "Excel":
                # Write Excel to zip
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer) as writer:
                    nifty_df.to_excel(writer, sheet_name="NIFTY")
                    vix_df.to_excel(writer, sheet_name="VIX")
                
                zip_file.writestr("nifty_vix_data.xlsx", excel_buffer.getvalue())
            
            elif export_format == "JSON":
                # Write JSONs to zip
                nifty_json = nifty_df.reset_index().to_json(orient="records", date_format="iso")
                vix_json = vix_df.reset_index().to_json(orient="records", date_format="iso")
                
                zip_file.writestr("nifty_data.json", nifty_json)
                zip_file.writestr("vix_data.json", vix_json)
        
        # Set response headers for download
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"nifty_vix_data_{timestamp}.zip"
        
        # Return as file download
        from flask import send_file
        zip_buffer.seek(0)
        return send_file(
            zip_buffer,
            mimetype="application/zip",
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        flash(f"Error exporting data: {e}", "error")
        return redirect(url_for('data.index'))

@bp.route('/import', methods=['POST'])
def import_data():
    """Import data from uploaded files"""
    try:
        # Check if files were uploaded
        if 'nifty_file' not in request.files or 'vix_file' not in request.files:
            flash("Both NIFTY and VIX files are required", "error")
            return redirect(url_for('data.index'))
        
        nifty_file = request.files['nifty_file']
        vix_file = request.files['vix_file']
        
        if nifty_file.filename == '' or vix_file.filename == '':
            flash("Both NIFTY and VIX files are required", "error")
            return redirect(url_for('data.index'))
        
        # Read uploaded files
        nifty_df = pd.read_csv(nifty_file, index_col=0, parse_dates=True)
        vix_df = pd.read_csv(vix_file, index_col=0, parse_dates=True)
        
        # Merge with existing data
        data_dir = 'data'
        nifty_consolidated_path = os.path.join(data_dir, 'nifty_data_consolidated.csv')
        vix_consolidated_path = os.path.join(data_dir, 'vix_data_consolidated.csv')
        
        if os.path.exists(nifty_consolidated_path) and os.path.exists(vix_consolidated_path):
            existing_nifty, existing_vix = load_from_csv(nifty_consolidated_path, vix_consolidated_path)
            
            # Merge
            nifty_df = merge_dataframes(existing_nifty, nifty_df)
            vix_df = merge_dataframes(existing_vix, vix_df)
        
        # Save merged data
        save_to_csv(nifty_df, vix_df, data_dir=data_dir)
        
        # Update session data summary
        data_summary = get_data_summary(data_dir=data_dir)
        session['data_summary'] = data_summary
        
        flash(f"Successfully imported data with {len(nifty_df)} NIFTY records and {len(vix_df)} VIX records", "success")
    except Exception as e:
        flash(f"Error importing data: {e}", "error")
    
    return redirect(url_for('data.index')) 