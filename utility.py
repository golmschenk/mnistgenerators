"""
Code for miscellaneous utility purposes.
"""
import os
import sys
import zipfile
import psutil


def log_source_files(output_file_name):
    """
    Takes all the Python and txt files in the working directory and compresses them into a zip file.

    :param output_file_name: The name of the output file.
    :type output_file_name: str
    """
    file_names_to_zip = []
    for file_name in os.listdir('.'):
        if file_name.endswith(".py") or file_name.endswith('.txt'):
            file_names_to_zip.append(file_name)
    with zipfile.ZipFile(output_file_name + '.zip', 'w') as zip_file:
        for file_name in file_names_to_zip:
            zip_file.write(file_name)


def make_this_process_low_priority():
    """Set the priority of the process to below normal."""
    try:
        sys.getwindowsversion()  # Errors to exception if not in Windows.
        current_process = psutil.Process(pid=os.getpid())
        current_process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except AttributeError:
        os.nice(1)
