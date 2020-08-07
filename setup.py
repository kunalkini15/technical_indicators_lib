import setuptools


with open('README.md') as f:
    README = f.read()

setuptools.setup(
    author="Kunal Kini K",
    author_email="kunalkini15@gmail.com",
    name='technical_indicators_lib',
    license="MIT",
    description='Technical Indicators Library provides means to derive stock market technical indicators.',
    version='0.0.1',
    long_description=README,
    url='https://github.com/kunalkini015/technical-indicators',
    packages=setuptools.find_packages(),
    python_requires=">=3.5",
    install_requires=['requests', 'pandas', 'numpy'],
    classifiers=[
        # Trove classifiers
        # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry'
    ],
)
