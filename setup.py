from distutils.core import setup


def readme():
    """Import the README.md Markdown file and try to convert it to RST format."""
    try:
        import pypandoc
        return pypandoc.convert('README.md', 'rst')
    except(IOError, ImportError):
        with open('README.md') as readme_file:
            return readme_file.read()


setup(
    name='ml-ids',
    version='0.1',
    description='Machine learning based Intrusion Detection System',
    long_description=readme(),
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    url='https://github.com/cstub/ml-ids',
    author='cstub',
    author_email='stumpf.christoph@gmail.com',
    license='MIT',
    packages=['ml_ids'],
    install_requires=[
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)
