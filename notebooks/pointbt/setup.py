import os
import re
from setuptools import setup, find_packages

# 0. pythonバージョンを指定
description = "a template for an original package" # need to update
python_ver = "3.10" # need to check

# 1. Read the requirements from the requirements.txt file
with open('requirements.txt') as requirements_file:
    install_requirements = requirements_file.read().splitlines()

# 2. Prepare the package metadata
# Helper functions
# Function to parse the filename and extract date and version
def parse_note_file(filename):
    match = re.match(r"cli_(\d{6})_(\d{2})\.py", filename)
    if match:
        date = int(match.group(1))  # Extract date as an integer
        version = int(match.group(2))  # Extract version as an integer
        return date, version
    return None

def find_top_level_packages():
    packages = []
    for root, dirs, files in os.walk("."):
        if "__init__.py" in files:
            relative_path = os.path.relpath(root, ".")
            package_name = relative_path.replace(os.sep, ".")
            if not any(package_name.startswith(pkg + ".") for pkg in packages):
                packages.append(package_name)
    return packages[0] # top level package

# Dynamically retrieve the version
def get_version():
    version = None
    package_dir = find_top_level_packages()
    init_file_path = os.path.join(package_dir, '__init__.py')
    if os.path.exists(init_file_path):
        with open(init_file_path) as f:
            content = f.read()
            match = re.search(r"^__version__ = ['\"]([^'\"]+)['\"]", content)
            if match:
                version = match.group(1)
    if version is None:
        version = "0.0.1"
    return version

# Dynamically retrieve the author
def get_author():
    author = None
    package_dir = find_top_level_packages()
    init_file_path = os.path.join(package_dir, '__init__.py')
    
    if os.path.exists(init_file_path):
        with open(init_file_path) as f:
            content = f.read()
            match = re.search(r"^__author__ = ['\"]([^'\"]+)['\"]", content)
            if match:
                author = match.group(1)
    if author is None:
        author = "Default Author"
    return author

# Find the package name dynamically
package_name = None
for subdir in os.listdir('.'):
    # Check if the subdir is a package (contains __init__.py)
    if os.path.isdir(subdir) and os.path.exists(os.path.join(subdir, '__init__.py')):
        package_name = subdir
        break
if package_name is None:
    raise FileNotFoundError("No valid package found in the current directory.")

# Find all Python files starting with "note_" and ending with ".py"
note_files = []
for f in os.listdir(os.path.join(os.path.dirname(__file__), package_name)):
    if f.startswith('cli_') and f.endswith('.py'):
        note_files.append(os.path.join(os.path.dirname(__file__), package_name, f))

# Parse the filenames and identify the latest file
latest_file = note_files[0] if note_files else None
if len(note_files) > 1:
    latest_date = -1
    highest_version = -1
    for note_file in note_files:
        parsed = parse_note_file(note_file)
        if parsed:
            date, version = parsed
            if date > latest_date or (date == latest_date and version > highest_version):
                latest_date = date
                highest_version = version
                latest_file = note_file
if latest_file:
    module_name = os.path.basename(latest_file).split('.')[0]  # Extract the module name without '.py'
else:
    raise FileNotFoundError("No valid 'cli_' file found in the current directory.")

# 3. Define the setup function
# modify entry_points to use command line if needed
# {COMMAND NAME}={module path}:{function in the module}
setup(
    name=f"{package_name}",
    version=f"{get_version()}",
    description=f"{description}",
    author=f"{get_author()}",
    packages=find_packages(),
    install_requires=install_requirements,
    entry_points={
        "console_scripts": [
            f"{package_name}={package_name}.{module_name}:main",
        ]
    },
    classifiers=[
        f'Programming Language :: Python :: {python_ver}', # need to check
    ]
)