import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clear_cache(data_dir):
    """
    Deletes cached files in the specified directory while preserving the source data file.
    
    Args:
        data_dir (str): Path to the data directory
    """
    # Define files to delete
    files_to_delete = [
        "bitcoin_procesado.csv",
        "bitcoin_discretizado.csv",
        "modelo_clustering.pkl",
        "modelo_anomalias.pkl",
        "reglas_asociacion.csv"
    ]
    
    # Convert data_dir to Path object
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.error(f"Directory does not exist: {data_path}")
        return
    
    # Iterate through files to delete
    for file_name in files_to_delete:
        file_path = data_path / file_name
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted: {file_path}")
            else:
                logger.info(f"File not found, skipping: {file_path}")
        except Exception as e:
            logger.error(f"Error deleting {file_path}: {str(e)}")

if __name__ == "__main__":
    # Define the data directory
    data_dir = r"c:\Users\maryc\OneDrive - Universidad de Guanajuato\Escritorio\Proyecto Mineria\datos"
    logger.info("Starting cache clearing process")
    clear_cache(data_dir)
    logger.info("Cache clearing process completed")