vsplit_factor=("0.8")
vuse_vars=(
"'l1pt','l1eta','l1phi','l2pt','l2eta','l2phi','l3pt','l3eta','l3phi','l4pt','l4eta','l4phi','j1pt','j1eta','j1phi','j2pt','j2eta','j2phi'"
"'l1pt','l1eta','l1phi','l2pt','l2eta','l2phi','l3pt','l3eta','l3phi','l4pt','l4eta','l4phi','j1pt','j1eta','j1phi','j2pt','j2eta','j2phi','j3pt','j3eta','j3phi'"
"'l1pt','l1eta','l1phi','l2pt','l2eta','l2phi','l3pt','l3eta','l3phi','l4pt','l4eta','l4phi','j1pt','j1eta','j1phi','j2pt','j2eta','j2phi','pfmet','njets','nbjets'"
"'l1pt','l1eta','l1phi','l2pt','l2eta','l2phi','l3pt','l3eta','l3phi','l4pt','l4eta','l4phi','j1pt','j1eta','j1phi','j2pt','j2eta','j2phi','j3pt','j3eta','j3phi','pfmet','njets','nbjets'"
"'l1pt','l1eta','l1phi','l2pt','l2eta','l2phi','l3pt','l3eta','l3phi','l4pt','l4eta','l4phi','j1pt','j1eta','j1phi','j1e','j2pt','j2eta','j2phi','j2e'"
"'l1pt','l1eta','l1phi','l2pt','l2eta','l2phi','l3pt','l3eta','l3phi','l4pt','l4eta','l4phi','j1pt','j1eta','j1phi','j1e','j2pt','j2eta','j2phi','j2e','j3pt','j3eta','j3phi','j3e'"
"'j1pt','j1eta','j1phi','j2pt','j2eta','j2phi'"
"'j1pt','j1eta','j1phi','j2pt','j2eta','j2phi','j3pt','j3eta','j3phi'"
"'l1pt','l1eta','l1phi','l2pt','l2eta','l2phi','l3pt','l3eta','l3phi','l4pt','l4eta','l4phi'"
"'z1mass','z2mass','thetastar','phistar1','theta1','theta2','phi'"
"'l1pt','l1eta','l1phi','l2pt','l2eta','l2phi','l3pt','l3eta','l3phi','l4pt','l4eta','l4phi','j1pt','j1eta','j1phi','j2pt','j2eta','j2phi','j3pt','j3eta','j3phi','j4pt','j4eta','j4phi'"
"'l1pt','l1eta','l1phi','l2pt','l2eta','l2phi','l3pt','l3eta','l3phi','l4pt','l4eta','l4phi','j1pt','j1eta','j1phi','j2pt','j2eta','j2phi','j3pt','j3eta','j3phi','j4pt','j4eta','j4phi','j5pt','j5eta','j5phi'"
"'l1pt','l1eta','l1phi','l2pt','l2eta','l2phi','l3pt','l3eta','l3phi','l4pt','l4eta','l4phi','j1pt','j1eta','j1phi','j2pt','j2eta','j2phi','j3pt','j3eta','j3phi','j4pt','j4eta','j4phi','j5pt','j5eta','j5phi','j6pt','j6eta','j6phi'"
)
vpre_process=("'no'" "'scale'" "'norm'")
vwait_for=("100" "500" "2400")
vsbatch=("128" "786")
vlrate=("-1")
vlayers=("7,5,3" "21,13,8" "30" "10,10,10,10" "100")
vweights=("'sum'" "'individual'")


rm Scripts/*

ijob=0
for spf in ${vsplit_factor[@]}
do
  for uv in ${vuse_vars[@]}
  do
    for pp in ${vpre_process[@]}
    do
      for wf in ${vwait_for[@]}
      do
	for sb in ${vsbatch[@]}
	do
	  for lr in ${vlrate[@]}
	  do
	    for ly in ${vlayers[@]}
	    do
	      for wt in ${vweights[@]}
	      do
	      
		((ijob+=1))
		#echo "----------- job $ijob ---------"
		#echo $spf
		#echo $uv
		#echo $pp
		#echo $wf
		#echo $sb
		#echo $lr
		#echo $ly
		#echo $wt

                if [ $ijob -lt 1981 ] || [ -e /eos/user/m/mmelodea/MonoHiggsHZZ4L/KerasNewScans2/logs/logrun$ijob.log ]
                then
                    echo "job$ijob concluded..."
                else
	   	    #first creates the analysis script
		    cat Needs/EventDNN_ControlSets.py > Scripts/EventDNN_ControlSets_$ijob.py
		    sed -i s/_split_factor/$spf/g Scripts/EventDNN_ControlSets_$ijob.py
		    sed -i s/_use_vars/$uv/g Scripts/EventDNN_ControlSets_$ijob.py 
		    sed -i s/_pre_process/$pp/g Scripts/EventDNN_ControlSets_$ijob.py 
		    sed -i s/_wait_for/$wf/g Scripts/EventDNN_ControlSets_$ijob.py 
		    sed -i s/_sbatch/$sb/g Scripts/EventDNN_ControlSets_$ijob.py 
		    sed -i s/_lrate/$lr/g Scripts/EventDNN_ControlSets_$ijob.py 
		    sed -i s/_layers/$ly/g Scripts/EventDNN_ControlSets_$ijob.py 
		    sed -i s/_use_weights/$wt/g Scripts/EventDNN_ControlSets_$ijob.py 
		
		    #then creates the jobs script
		    echo "cd /afs/cern.ch/work/m/mmelodea/private/KerasJobs" >> Scripts/keras_job$ijob.job
                    echo "mkdir /tmp/mmelodea/keras$ijob" >> Scripts/keras_job$ijob.job
		    echo "cp Scripts/EventDNN_ControlSets_$ijob.py /tmp/mmelodea/keras$ijob" >> Scripts/keras_job$ijob.job
		    echo "cp Needs/UserFunctions.py /tmp/mmelodea/keras$ijob" >> Scripts/keras_job$ijob.job
		    echo "cp Needs/hzz4l_vbf_selection_noDjet_m4l118-130GeV_shuffledFS.pkl /tmp/mmelodea/keras$ijob" >> Scripts/keras_job$ijob.job
		    echo "cd /tmp/mmelodea/keras$ijob" >> Scripts/keras_job$ijob.job
		    echo "source ~/lcgenv.sh" >> Scripts/keras_job$ijob.job
		    echo "source ~/keras/bin/activate" >> Scripts/keras_job$ijob.job
		    echo "python EventDNN_ControlSets_$ijob.py > logrun.log" >> Scripts/keras_job$ijob.job
		    echo "cp logrun.log /eos/user/m/mmelodea/MonoHiggsHZZ4L/KerasNewScans2/logs/logrun$ijob.log" >> Scripts/keras_job$ijob.job
		    echo "cp best_weights.hdf5 /eos/user/m/mmelodea/MonoHiggsHZZ4L/KerasNewScans2/weights/best_weights$ijob.hdf5" >> Scripts/keras_job$ijob.job
		    echo "cp ROCsComparisonTrainTest.png /eos/user/m/mmelodea/MonoHiggsHZZ4L/KerasNewScans2/rocs_train_test/ROCsComparisonTrainTest$ijob.png" >> Scripts/keras_job$ijob.job
		    echo "cp ROCsComparisonDiscriminants.png /eos/user/m/mmelodea/MonoHiggsHZZ4L/KerasNewScans2/rocs_discriminants/ROCsComparisonDiscriminants$ijob.png" >> Scripts/keras_job$ijob.job
		    echo "cp MetricsComparisonDiscriminants.png /eos/user/m/mmelodea/MonoHiggsHZZ4L/KerasNewScans2/metrics/MetricsComparisonDiscriminants$ijob.png" >> Scripts/keras_job$ijob.job
		    echo "cp DiscriminantsDistributions.png /eos/user/m/mmelodea/MonoHiggsHZZ4L/KerasNewScans2/distributions/DiscriminantsDistributions$ijob.png" >> Scripts/keras_job$ijob.job
		    echo "cp SeparatedROCsComparisonMELAtoNetwork.png /eos/user/m/mmelodea/MonoHiggsHZZ4L/KerasNewScans2/separated_rocs/SeparatedROCsComparisonMELAtoNetwork$ijob.png" >> Scripts/keras_job$ijob.job
		    echo "cp TrainingHistory.png /eos/user/m/mmelodea/MonoHiggsHZZ4L/KerasNewScans2/train_history/TrainingHistory$ijob.png" >> Scripts/keras_job$ijob.job
		    #echo "rm ./*" >> Scripts/keras_job$ijob.job
		
		    #and finaly runs each job
		    echo "bsub -q 2nd -J job$ijob < Scripts/keras_job$ijob.job"
		    bsub -q 2nd -J job$ijob < Scripts/keras_job$ijob.job
		fi
	      done
	    done
	  done
	done
      done
    done
  done
done
