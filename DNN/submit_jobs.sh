#### script to submit multiple NN training jobs at lxplus
#### Author: Miqueias Melo de Almeida
#### Date: 06/06/2018

vnninputs2=( 
  "f_lept1_pt f_lept1_eta f_lept1_phi f_lept2_pt f_lept2_eta f_lept2_phi f_lept3_pt f_lept3_eta f_lept3_phi f_lept4_pt f_lept4_eta f_lept4_phi f_jets_highpt_pt[0] f_jets_highpt_eta[0] f_jets_highpt_phi[0] f_jets_highpt_pt[1] f_jets_highpt_eta[1] f_jets_highpt_phi[1]"
  #"f_lept1_pt f_lept1_eta f_lept1_phi f_lept2_pt f_lept2_eta f_lept2_phi f_lept3_pt f_lept3_eta f_lept3_phi f_lept4_pt f_lept4_eta f_lept4_phi f_jets_highpt_pt[0] f_jets_highpt_eta[0] f_jets_highpt_phi[0] f_jets_highpt_e[0] f_jets_highpt_pt[1] f_jets_highpt_eta[1] f_jets_highpt_phi[1] f_jets_highpt_e[1]"
  "f_lept1_pt f_lept1_eta f_lept1_phi f_lept2_pt f_lept2_eta f_lept2_phi f_lept3_pt f_lept3_eta f_lept3_phi f_lept4_pt f_lept4_eta f_lept4_phi f_jets_highpt_pt[0] f_jets_highpt_eta[0] f_jets_highpt_phi[0] f_jets_highpt_pt[1] f_jets_highpt_eta[1] f_jets_highpt_phi[1] f_pfmet"
  #"f_lept1_pt f_lept1_eta f_lept1_phi f_lept2_pt f_lept2_eta f_lept2_phi f_lept3_pt f_lept3_eta f_lept3_phi f_lept4_pt f_lept4_eta f_lept4_phi f_jets_highpt_pt[0] f_jets_highpt_eta[0] f_jets_highpt_phi[0] f_jets_highpt_e[0] f_jets_highpt_pt[1] f_jets_highpt_eta[1] f_jets_highpt_phi[1] f_jets_highpt_e[1] f_pfmet"
) 

vnninputs3=(
  "f_lept1_pt f_lept1_eta f_lept1_phi f_lept2_pt f_lept2_eta f_lept2_phi f_lept3_pt f_lept3_eta f_lept3_phi f_lept4_pt f_lept4_eta f_lept4_phi f_jets_highpt_pt[0] f_jets_highpt_eta[0] f_jets_highpt_phi[0] f_jets_highpt_pt[1] f_jets_highpt_eta[1] f_jets_highpt_phi[1] f_jets_highpt_pt[2] f_jets_highpt_eta[2] f_jets_highpt_phi[2]"
  #"f_lept1_pt f_lept1_eta f_lept1_phi f_lept2_pt f_lept2_eta f_lept2_phi f_lept3_pt f_lept3_eta f_lept3_phi f_lept4_pt f_lept4_eta f_lept4_phi f_jets_highpt_pt[0] f_jets_highpt_eta[0] f_jets_highpt_phi[0] f_jets_highpt_e[0] f_jets_highpt_pt[1] f_jets_highpt_eta[1] f_jets_highpt_phi[1] f_jets_highpt_e[1] f_jets_highpt_pt[2] f_jets_highpt_eta[2] f_jets_highpt_phi[2] f_jets_highpt_e[2]"
  "f_lept1_pt f_lept1_eta f_lept1_phi f_lept2_pt f_lept2_eta f_lept2_phi f_lept3_pt f_lept3_eta f_lept3_phi f_lept4_pt f_lept4_eta f_lept4_phi f_jets_highpt_pt[0] f_jets_highpt_eta[0] f_jets_highpt_phi[0] f_jets_highpt_pt[1] f_jets_highpt_eta[1] f_jets_highpt_phi[1] f_jets_highpt_pt[2] f_jets_highpt_eta[2] f_jets_highpt_phi[2] f_pfmet"
  #"f_lept1_pt f_lept1_eta f_lept1_phi f_lept2_pt f_lept2_eta f_lept2_phi f_lept3_pt f_lept3_eta f_lept3_phi f_lept4_pt f_lept4_eta f_lept4_phi f_jets_highpt_pt[0] f_jets_highpt_eta[0] f_jets_highpt_phi[0] f_jets_highpt_e[0] f_jets_highpt_pt[1] f_jets_highpt_eta[1] f_jets_highpt_phi[1] f_jets_highpt_e[1] f_jets_highpt_pt[2] f_jets_highpt_eta[2] f_jets_highpt_phi[2] f_jets_highpt_e[2] f_pfmet"
)

vminimizer=("sgd" "adam" "adagrad" "adadelta" "rmsprop")
vneuron=("relu" "selu")
vpatiences=("10" "100" "300")
vbatches=("32" "128")
vtopologies=("21 13 8" "30" "100" "11 7 3 6")
vweights=("mc_weight" "event_weight")
voutliers=("" "--nooutliers")


for njet in 2 3
do
  vnninputs=()
  if [ $njet -eq 2 ]
  then
    vnninputs=("${vnninputs2[@]}")
  else
    vnninputs=("${vnninputs3[@]}")
  fi

  scheme=0
  for outliers in "${voutliers[@]}"
  do
    for topology in "${vtopologies[@]}"
    do
      for batch in "${vbatches[@]}"
      do
        for patience in "${vpatiences[@]}"
        do
	  for weight in "${vweights[@]}"
	  do
            for nninputs in "${vnninputs[@]}"
            do
              #echo "Inputs: ${nninputs}"
              for neuron in "${vneuron[@]}"
              do
                for minimizer in "${vminimizer[@]}"
                do
                  if [ "$minimizer" != "adam" ]
                  then
                    if [ -e Njets${njet}/Model${scheme}/results/SummaryOfResults.pkl ]
                    then
                      echo "Njets${njet}/Model${scheme} is ready!"
                    else
                      echo "cd /afs/cern.ch/work/m/mmelodea/private/KerasJobs/Trainings/V5" > k${scheme}nj${njet}.job
                      echo "mkdir -p Njets${njet}/Model${scheme}" >> k${scheme}nj${njet}.job
                      echo "cp EvaluateNeuralNetwork.py Njets${njet}/Model${scheme}/" >> k${scheme}nj${njet}.job
                      echo "cd Njets${njet}/Model${scheme}/" >> k${scheme}nj${njet}.job
                      echo "source /afs/cern.ch/user/m/mmelodea/loadKeras.sh" >> k${scheme}nj${njet}.job
                      echo '#Run the training' >> k${scheme}nj${njet}.job
                      echo "python EvaluateNeuralNetwork.py --infile /afs/cern.ch/work/m/mmelodea/private/KerasJobs/Trainings/V5/Samples/hzz4l_vbf_selection_m4l118-130GeV_shuffledFS_njets${njet}.pkl --keys qqH ggH ggZZ ZZJJ qqZZ ttH HWW TTV WH ZH --signal qqH --nninputs ${nninputs} --topology ${topology} --batchsize ${batch} --patience ${patience} --scaletrain ${weight} --split 0.8 --nepochs 10000 --neuron ${neuron} --minimizer ${minimizer} ${outliers} | tee training.log" >> k${scheme}nj${njet}.job
                      ##then submit the job
                      echo "bsub -J v5k${scheme}nj${njet} -q 8nh < k${scheme}nj${njet}.job"
                      bsub -J v5k${scheme}nj${njet} -q 8nh < k${scheme}nj${njet}.job
                    fi
                  fi
	          scheme=$((scheme+1))
                done
              done
            done
          done
	done
      done
    done
  done
done
