# VM instructions from Lecture 10 & 11
This guide is a note scribing guide from Lecture 10 and first 30 min from Lecture 11, which intends to provide simple and clear instructions on how to parse DEEP data and backtest on virtual machine. First section [Transfer file between VM and Local Computer](#transfer-file-between-vm-and-local-computer) is intended to provide instructions on the transfer between VM and local computer, including using ssh remote control. Second section [Backtesting](#backtesting) is intended to provide instructions on how to parse DEEP data and how to do backtest on VM. Third section [Backtesting in one command](#backtesting-in-one-command) covers the first 30 min from Lecture 11, and provides a simplified way to backtest and transfer files that  functions similarly as the first two parts. This part is an independent part that needs to use a new VM and starts over everthing in the first two parts again.

## Transfer file between VM and Local Computer
### From VM to Local Computer
1. [Not necessary] Open VM, on top bar, find "Machine" -> "Settings" -> "Network" -> "Advanced", click "Port Forwarding". Your Host IP 127.0.0.1, Host Port 2222, Guest Port 22 can be found there.
2. Open your local computer terminal, type `ssh -p 2222 vagrant@127.0.0.1`. Answer necessary questions. Note that here you may use keys to avoid authentication, as mentioned in [Generate Keys](#generate-keys) section.
3. Your path has changed to vagrant now. Change directory to the file you want to copy to your local computer.
Type `pwd`, this is the current directory you will need in next step.
4. Open a new terminal window in local computer, change directory to the folder you want to copy to. Type `scp -P 2222 vagrant@127.0.0.1:/home/vagrant/samplepath/sample.txt .`. Remove the samplepath and sample.txt by the path and file name you got from last step.
### From Local Computer to VM
Change directory to the file you want to copy to VM in local computer, say, test.txt. Syntax is `scp -P 2222 test.txt vagrant@127.0.0.1:~/Desktop/test.txt`. In this way, test.txt will be copied to Desktop in VM.

### Generate Keys
To generate keys for local computer to visit VM, in local computer terminal, type:
1. `scp -P 2222 vagrant@127.0.0.1:~/.ssh/authorized_keys .` 
2. `ssh-copy-id -p 2222 vagrant@127.0.0.1`

To generate keys for VM to visit gitlab, in vagrant terminal, type:
1. `ls -lA ~/.ssh`
2. `ssh-keygen -t rsa -b 4096`
 and hit enter till the end.
 3. `cat ~/.ssh/id_rsa.pub` 
 4. copy the keys, which is the output from last step, to your gitlab account. Click Profile photo -> "Preferences" -> "SSH Keys", and paste. Click "Add key".

### Notes
It's not always necessary to change path to vagrant. You can run commands in local computer like `ssh -p 2222 vagrant@127.0.0.1 'ls -lA' ` to log into vagrant and run `ls -lA`.

## Backtesting
### Get Sample Strategy From Gitlab
In VM, do:
1. Open Browser -> Gitlab -> example_projects -> DiaIndexArb -> Clone -> Clone with SSH. Copy URL.
2. Open Terminal. Type `cd Desktop/strategy_studio/localdev/RCM/StrategyStudio/examples/strategies/`.
3. `git clone URL`. Paste URL from step 1.

### Open Eclipse
Open Eclipse -> using default workspace path and launch -> import projects -> General -> Projects from Folder or Archive -> Next -> Directory -> Find DiaIndexArb -> click Open -> click Finish.

### Preparations
1. Git is useful to keep track of all the changes in case needed. This is not required, though. **Do not git pull, or you will get sued!!! Keep them locally!!!**
If you are interested in the details about how to use git to keep track of changes, see [Lecture 10](https://davidl.web.engr.illinois.edu/fin566_fall_2021/) 56:40.
2. If the license.txt is not in the <em>backtesting</em> folder, check HW2 for details and add license.txt to <em>backtesting</em> folder.
3. In terminal, cd to <em>backtesting</em> folder. `nano backtester_config.txt` to ensure DISABLE_COMMAND_LINE=false and uncheck TEXT_TICK_COMPRESSED=true. Note that removing the # before TEXT_TICK_COMPRESSED is not asked in previous homework, and this step is to ensure that strategy studio will find .txt.gz file instead of .txt file.

### Parse DEEP data
1. Download DEEP data from [IEX](https://iextrading.com/trading/market-data/). If a large hard drive is available, use [download_iex_pcaps.py](https://gitlab.engr.illinois.edu/fin566_algo_market_micro_fall_2021/example_projects/iexdownloaderparser/-/blob/main/src/download_iex_pcaps.py) can automatically download data from IEX. 
2. In VM, clone parser project from [gitlab shared_code group](https://gitlab-beta.engr.illinois.edu/shared_code). `git clone git@gitlab.engr.illinois.edu:shared_code/strategystudiovagrantvm.git`.
3. Install pypy and other modules. 
    * Download PyPy3.7, Linux x86 64 bit from [here](https://www.pypy.org/download.html).
    * Install PyPy. Use command `tar xf Downloads/pypy3.7-v7.3.7-linux64.tar.bz2`.
    * Install other modules: 
    ```
    /home/vagrant/pypy3.7-v7.3.7-linux64/bin/pypy -m ensurepip
    ./pypy3.7-v7.3.7-linux64/bin/pypy -mpip install requests
    ./pypy3.7-v7.3.7-linux64/bin/pypy -mpip install tqdm
    ./pypy3.7-v7.3.7-linux64/bin/pypy -mpip install pytz
    sudo yum install tcpdump
    ```
4. Parse data. In terminal, change directory to iexdownloaderparser. Change directory / file name when necessary and run
```
gunzip -d -c ~/Downloads/IEX_deep/data_feeds_20211029_20211029_IEXTP1_DEEP1.0.pcap.gz | tcpdump -r - -w - -s 0 | ~/pypy3.7-v7.3.7-linux64/bin/pypy ~/Downloads/strategystudiovagrantvm/iexdownloaderparser/src/parse_iex_pcap.py /dev/stdin --symbols SPY,DIA,UNH,GS,HD,MSFT,CRM,MCD,HON,BA,V,AMGN,CAT,MMM,NKE,AXP,DIS,JPM,JNJ,TRV,AAPL,WMT,PG,IBM,CVX,MRK,DOW,CSCO,KO,VZ,INTC,WBA,KD
```
5. If everything works well, you should be able to see tick_TICKERNAME_DATE.txt.gz in iexdownloaderparser/data/text_tick_data.

### Backtest
1. Move all the tick_TICKERNAME_DATE.txt.gz in iexdownloaderparser/data/text_tick_data to strategy_studio/backtesting/text_tick_data.
```
cp -R ~/Downloads/strategystudiovagrantvm/iexdownloaderparser/data/text_tick_data/tick_*.txt.gz ~/Desktop/strategy_studio/backtesting/text_tick_data
```
2. [Open Eclipse](#open-eclipse). 
    * [Not necessary] Pull "Makefile" in the left column. Check the following code:
    ```
    all: $(HEADERS) $(LIBRARY)
    ...
    copy_strategy: all
	cp $(LIBRARY) /home/vagrant/Desktop/strategy_studio/backtesting/strategies_dlls/.
    ```
    This is to make sure all is done before copy_strategy. "all" here requires header file and library, which is to make sure you build the latest version.

    * On left column, right click SimpleTradeStrategy[DialndexArb master] -> Build Targets -> Create -> Target name: copy_strategy. Build copy_strategy.
3. cd to Desktop/strategy_studio/backtesting. <!--Before backtest, in strategies_dlls folder, remove any *.so other than DiaIndexArb.so file. -->
Run `./StrategyServerBacktesting`. Make sure you can find "Loading strategy dll /home/vagrant/Desktop/strategy_studio/backtesting/strategies_dlls/DiaIndexArb.so..." in output.
4. In a new vagrant terminal, cd to Desktop/strategy_studio/backtesting/utilities. "StrategyCommandLine" is to programmatically send commands into the command-line interface opened in step 3. 
    * Copy ` cp cmd_config_example.txt cmd_config.txt` and edit `nano cmd_config.txt`. Change `USERNAME=username` to `USERNAME=dlariviere`. Save the change. (Reference: Lecture 6, 1:10:31)
    * Run `./StrategyCommandLine help`. We will use Single command mode. Run `./StrategyCommandLine cmd strategy_instance_list
` to test if you can run command "strategy_instance_list" as the same way you could in the command-line interface. 
5. To create instance: a sample command to run in the command-line interface in binary StrategyServerBacktesting is 
```
create_instance TestOneDiaIndexArbStrategy DiaIndexArbStrategy UIUC SIM-1001-101 dlariviere 1000000 -symbols DIA|UNH|GS|HD|MSFT|CRM|MCD|HON|BA|V|AMGN|CAT|MMM|NKE|AXP|DIS|JPM|JNJ|TRV|AAPL|WMT|PG|IBM|CVX|MRK|DOW|CSCO|KO|VZ|INTC|WBA|KD
```
This command can be found in eclipse -> commands.txt. 

6. To backtest: 
    * Notice a makefile target called "launch_backtest". Build targets as in step 2 with Target name "launch_backtest", which is meant to run "start_backtest 2021-11-05 2021-11-05 TestOneDiaIndexArbStrategy 1" in the command-line interface. Here "1" means only backtest on trades. Now we have book data, "1" could be changed to "0".
    * Run launch_backtest. 

7. After making modifications on code:
    * Run copy_strategy. This would automatically rebuild the strategy.
    * Relaunch `./StrategyServerBacktesting` in terminal. Go back to eclipse and run launch_backtest.




## Backtesting in one command


1. Download and install [vagrant](https://www.vagrantup.com/downloads) for your local computer.
2. Git clone [StrategyStudioVagrantVM](https://gitlab.engr.illinois.edu/fin566_algo_market_micro_fall_2021/example_projects/strategystudiovagrantvm) folder from GitLab: `git clone git@gitlab.engr.illinois.edu:fin566_algo_market_micro_fall_2021/example_projects/strategystudiovagrantvm.git`.
3. Download [marketdata folder](https://uofi.app.box.com/folder/150244581567) and [Fin566BaseVagrantVM.box
](https://uofi.app.box.com/file/885625113867) from illinois box > All Files > fall_2021_strategy_studio_vagrant_vm to StrategyStudioVagrantVM folder. Put license.txt you got from RCM into strategystudiovagrantvm/backup_files folder.
4. In Vagrantfile, change `vb.customize ["modifyvm", :id, "--cpus", "4"]` to `vb.customize ["modifyvm", :id, "--cpus", "2"]`. This is the number of CPUs when VM is created, and should match your license file -> NUM CPUS. 
5. cd to strategystudiovagrantvm, run `./go.sh`. If successful, backtest results should be seen.
6. To access VM, in local terminal,  cd to strategystudiovagrantvm and run `vagrant ssh`. Necessary changes on algo can be made in, say, `Desktop/strategy_studio/localdev/RCM/StrategyStudio/examples/strategies/DiaIndexArb/DiaIndexArb.cpp`.
7. After making modifications to code:
    * In VM, cd to DiaIndexArb and run `make copy_strategy`. 
    * In local machine cd to strategystudiovagrantvm, run `vagrant ssh -c'/vagrant/provision_scripts/run_backtest.sh'` to backtest again.
8. Via shared folder, it's easy to trasfer file between host and VM. The file in VM home -> cd /vagrant and file in local machine cd strategystudiovagrantvm automatically matches.

## Appendix - Parsed data format interpretation

The data parsed in [Parse DEEP data](#parse-deep-data) section has the format tick_TICKERNAME_DATE.txt.gz. Unzip this file, you will get a tick_TICKERNAME_DATE.txt file. A couple of lines as an example is listed below:
```
2021-10-29 13:18:34.129483008,2021-10-29 13:18:34.129468073,188453,T,IEX,206.200000,25
2021-10-29 13:20:10.563963904,2021-10-29 13:20:10.563937021,194503,T,IEX,206.930000,25
2021-10-29 13:21:24.27126016,2021-10-29 13:21:24.27103578,199540,T,IEX,206.810000,25
2021-10-29 13:30:00.963740928,2021-10-29 13:30:00.963649251,279062,P,IEX,1,207.710000,33,,,,0
2021-10-29 13:30:00.968321024,2021-10-29 13:30:00.968302715,279175,P,IEX,1,207.710000,0,,,,1
2021-10-29 13:30:00.968321024,2021-10-29 13:30:00.968302715,279176,P,IEX,1,207.700000,33,,,,0
```
For "T" type trade data, use first line as an example:
* 2021-10-29 13:18:34.129483008: collection_time_string, packet capture time in nanoseconds.
* 2021-10-29 13:18:34.129468073: source_time_string, send time.
* 188453: seq_num, which is message id.
* 206.200000: price.
* 25: size.

For "P" type auction data, use 4th line as an example:
* 2021-10-29 13:30:00.963740928: collection_time_string, packet capture time in nanoseconds.
* 2021-10-29 13:30:00.963649251: source_time_string, send time.
* 279062: seq_num, which is message id.
* 1: side_int. 1 means BID, 2 means ASK.
* 207.710000: price.
* 33: size.
* 0: is_partial_int. 1 means this is a partial event, 0 means event processing completed.

Details can be found in [parse_iex_pcap.py](https://gitlab.engr.illinois.edu/fin566_algo_market_micro_fall_2021/example_projects/iexdownloaderparser/-/blob/main/src/parse_iex_pcap.py).



