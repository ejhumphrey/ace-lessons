from setuptools import setup

long_description = \
    """Automatic Chord Estimation tools for visualizing and wrangling
    results, accompanying Humphrey, E. & Bello, J. P., 'Four Timely Lessons on
    Automatic Chord Estimation.' Proceedings of the 16th ISMIR, 2015."""

setup(
    name='ace_tools',
    version='1.0.0',
    description='',
    author='Eric J. Humphrey',
    author_email='ejhumphrey@nyu.edu',
    url='http://github.com/ejhumphrey/ace-lessons',
    download_url='http://github.com/ejhumphrey/ace-lessons/releases',
    packages=['ace_tools'],
    package_data={},
    long_description=long_description,
    classifiers=[
        "License :: OSI Approved :: ISC License (ISCL)",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7"
    ],
    keywords='',
    license='ISC',
    install_requires=[
        'numpy >= 1.9.0',
        'jams == 0.1',
        'mir_eval == 0.1',
        'matplotlib',
        'seaborn',
        'mpld3'
    ]
)
