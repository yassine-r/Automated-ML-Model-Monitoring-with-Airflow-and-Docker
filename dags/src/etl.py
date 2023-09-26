
import os
import shutil
import datetime
import sys
import config
import helpers


def get_latest_file(files):
  """Gets the name of the latest file in a list of files.

  Args:
    files: A list of files.

  Returns:
    The name of the latest file.
  """

  latest_file = None
  latest_date = None
  for file in files:
    file_date = datetime.datetime.strptime(file[:10], '%Y_%m_%d')
    if latest_date is None or file_date > latest_date:
      latest_file = file
      latest_date = file_date

  return latest_file



def collect_data(job_id: str):
    """
    Collects new data for processing based on the provided job_id.

    This function is designed to gather fresh data, typically stored in the directory
    `data/source`. The origin of this data could vary, including databases
    or other sources. The crucial aspect is that with each execution of the code,
    the most recently added data will be duplicated into the directory `data/collected`
    and will be renamed according to the provided job_id.

    Parameters:
    job_id (str): An identifier for the job. 

    Notes:
    - The assumption is made that the data in `data/source` is regularly
      updated with new information.
    """

    source_dir = os.path.join(config.PATH_DIR_DATA, 'source')
    collected_dir = os.path.join(config.PATH_DIR_DATA, 'collected')

    # Get the most recently added file from the source directory.
    latest_source_file = get_latest_file(os.listdir(source_dir))
    latest_source_file_path = os.path.join(source_dir, latest_source_file)

    # Create a new file in the collected directory with the job_id as the filename.
    new_file = os.path.join(collected_dir, f'{job_id}.csv')

    # Copy the contents of the latest source file to the new file.
    shutil.copy2(latest_source_file_path, new_file)

    # Print a message indicating that the data collection was successful.
    print(f'[INFO] Data collection for job {job_id} successful.')

    return new_file

if __name__=="__main__":
    job_id = helpers.generate_uuid()
    collect_data(job_id)