#for i in $(seq 0.1 0.1 0.5);
for i in $(seq 0.1 0.1 0.5);
do
	#python3 run.py -b "MovieLens 10M" "Good Books" "Yahoo Music" -m Random MostPopular UCB ThompsonSampling LinearEGreedy LinearUCB GLM_UCB WSPB --num_tasks 4 --evaluation_policies_parameters.LimitedInteraction.recommend_test_data_rate_limit=$i
	echo Running $i
	python3 run.py -b "MovieLens 10M" "Good Books" "Yahoo Music" -m Random MostPopular LinearUCB GLM_UCB WSPB --num_tasks 5 --evaluation_policies_parameters.LimitedInteraction.recommend_test_data_rate_limit=$i --defaults.interactors_evaluation_policy=LimitedInteraction
	#python3 run.py -b "MovieLens 10M" -m Random MostPopular LinearUCB GLM_UCB WSPB --num_tasks 2 --evaluation_policies_parameters.LimitedInteraction.recommend_test_data_rate_limit=$i
	#python3 run.py -b "MovieLens 10M" -m WSPB --num_tasks 2 --evaluation_policies_parameters.LimitedInteraction.recommend_test_data_rate_limit=$i
	#python3 run.py -b "MovieLens 10M" "Good Books" "Yahoo Music" -m LinearUCB GLM_UCB WSPB --num_tasks 4 --evaluation_policies_parameters.LimitedInteraction.recommend_test_data_rate_limit=$i
	sleep 1
done
