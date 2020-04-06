First export these variables when running the interactors to desactive the numpy's threads to reduce the overhead of multiple thread for the processes.
::
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1                                       
    export OMP_NUM_THREADS=1

To execute the code install the requirements

::
   pip install -r requirements.txt
