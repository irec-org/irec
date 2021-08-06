for i in $(seq 0.1 0.1 0.5);
do
	#python3 run.py -b "MovieLens 10M" "Good Books" "Yahoo Music" -m Random MostPopular UCB ThompsonSampling LinearEGreedy LinearUCB GLM_UCB OurMethod2 --num_tasks 4 --evaluation_policies_parameters.LimitedInteraction.recommend_test_data_rate_limit=$i
	#python3 run.py -b "Good Books" "Yahoo Music" -m Random MostPopular LinearUCB GLM_UCB OurMethod2 --num_tasks 5 --evaluation_policies_parameters.LimitedInteraction.recommend_test_data_rate_limit=$i&
	#python3 run.py -b "MovieLens 10M" -m Random MostPopular LinearUCB GLM_UCB OurMethod2 --num_tasks 2 --evaluation_policies_parameters.LimitedInteraction.recommend_test_data_rate_limit=$i
	#python3 run.py -b "MovieLens 10M" -m OurMethod2 --num_tasks 2 --evaluation_policies_parameters.LimitedInteraction.recommend_test_data_rate_limit=$i
	python3 run.py -b "MovieLens 10M" "Good Books" "Yahoo Music" -m LinearUCB GLM_UCB OurMethod2 --num_tasks 4 --evaluation_policies_parameters.LimitedInteraction.recommend_test_data_rate_limit=$i
done
