# VHcc
Search for VH(cc) process with CMS data, using coffea processor

## Software setup
 * First follow the [Readme instructions from CoffeaRunner repo](https://github.com/cms-rwth/CoffeaRunner/blob/master/README.md).
 * Then, checkout this repo into `./src/` directory:
```
git clone git@github.com:cms-rwth/VHcc src/VHcc
```
 * Re-compile: 
 `pip install -e .`


## Example job submission for VHcc

 * Run the code using a config file:
```
python runner_wconfig.py --cfg src/VHcc/cfg_VHcc.py
```
 * The processor where the selection is implemented is found at `src/VHcc/workflows/Zll_process_newHist.py`
 * The config file - `src/VHcc/cfg_VHcc.py` - governs the submission parameters. For example:
   * `dataset` - the names of the datasets to run over, including the path to .json file obtained with fetcher.
     * Using `samples` and `samples_exclude` parameters one can run over specific samples.
   * `workflow` - the worflow to run
   * `run_options` - jobs submission parameters
   * `userconfig` - one can also create dedicated parameters for specific workflos.
   * More details on the config can be found in [CoffeRunner's Readme](//github.com/cms-rwth/CoffeaRunner/blob/master/README.md) 


There are other processors one can run from this repository:
 * `cfg_genZjets.py` - generator level study of various Z+jets samples
 * `cfg_recoZjets_ULXX.py` - reco level selectin of Z(LL)+2jets running on various Z+jets samples

### Useful commands to remove jobs from condor
#### Why?
To not fill the queue with more than the allowed number of individual jobs. (5000 I think)

(Because parsl does not delete finished jobs from the queue!)

#### How?
Remove successfully finished job (changes status to removed)
```
condor_rm --constraint 'JobStatus==4'
```
Actually force deletion of removed jobs from queue after being removed
```
condor_rm -forcex --constraint 'JobStatus==3'
```
If there are other jobs (unrelated) that should not be touched by this, there is also the option to remove a range of jobs.
```
condor_rm -forcex --constraint 'ClusterId > 21849531 && ClusterId < 21849551'
```