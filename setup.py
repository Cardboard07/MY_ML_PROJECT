from setuptools import find_packages,setup

def get_requirements(file_path:str)->list[str]:
    requirements=[]
    with open(file_path) as f:
        requirements=f.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')

setup(
    name="First_ML",
    version='0.0.1',
    author='hrishi',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)