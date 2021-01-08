from setuptools import setup, find_packages

# find the directory and work backwards 1 level to find readme

setup(
    name="my_k_means",

    version='0.2.8',

    license='None',

    author='James Rose',

    author_email='JamesERose8@gmail.com',

    description=(
        'A program that can cluster vectors using k means'
        ' strategy to group likewise data.'),

    classifiers=[
        'Development status :: 0 - Pre-alpha',

        'Intended Audience :: Data Engineers, Scientists, Software Engineers',

        'Topic :: Software Development :: Libraries',

        'Programming Language :: Python - Anaconda :: 3.8.5'
    ],

    packages=find_packages(),

    long_description=open('C:\\Users\\METIC\\source\\repos\\KMeansClusteringPython\\README.md').read(),

    zip_safe=False,

    setup_requires=[],

    test_suite='')