from setuptools import setup, find_packages

setup(
    name="fred_data_fetch",  # Name of your package
    version="0.1.0",  # Version of your package
    author="Roberto Gomes",  # Your name
    author_email="roberto.gomes@gmail.com",  # Your email
    description="A package to fetch macroeconomic data from FRED",  # Short description
    long_description=open("README.md").read(),  # Long description from README.md
    long_description_content_type="text/markdown",  # Type of long description
    url="https://github.com/rgg81/fundamental-trading-signals",  # URL to your project repository
    package_dir={"": "src"},  # Specify that packages are located under the `src` directory
    packages=find_packages(where="src"),  # Find all packages in the `src` directory
    install_requires=[  # List of dependencies
        "pandas",
        "fredapi",
        "requests",
        "pyjstat",
    ],
    python_requires=">=3.11",  # Minimum Python version required
    classifiers=[  # Metadata about your package
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)