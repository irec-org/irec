for i in $(seq 0.1 0.1 0.5);
do
	python3 run.py -b "MovieLens 10M" "Good Books" "Yahoo Music" -m Random MostPopular UCB ThompsonSampling LinearEGreedy LinearUCB GLM_UCB OurMethod2 --num_tasks 2
done
