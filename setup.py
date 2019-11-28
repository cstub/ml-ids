from distutils.core import setup

setup(
    name='ml-ids',
    version='0.1',
    description='Machine learning based Intrusion Detection System',
    long_description='Machine learning based Intrusion Detection System',
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
