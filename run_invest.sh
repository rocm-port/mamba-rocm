## Use "source run_invest.sh" to run this.
conda activate py_3.9
./reset_env.sh
python setup.py install > install_log.txt
python bwd_investigation.py > invest_results.txt