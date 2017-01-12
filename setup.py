from setuptools import setup

setup(
    name='discourspublic',
    version='0.1',
    py_modules=['discourspublic'],
    install_requires=[],
    entry_points='''
        [console_scripts]
        discourspublic=discourspublic:cli
    ''',
)