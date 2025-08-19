# 1.1
Let $E_t$ denote the event that $\pi_{\theta}(s_t) \ne \pi^\star(s_t)$  
Then we have $\frac{1}{2}\sum_{s_t}|p_{\pi_\theta(s_t)}-p_{\pi^\star(s_t)}| \le Pr(\cup_iE_i) \le \sum_iPr(E_i) \le T\epsilon$  
# 1.2
## (a)
With the conclusion that $Pr(\cup_iE_i) \le T\epsilon$, we can say that at the end the probability that the learned policy $\pi_\theta$ makes a mistake at least once is at most $T\epsilon$.  
Thus,  $J(\pi^\star) - J(\pi_\theta) \le 2T\epsilon R_{max}$
## (b)
$J(\pi^\star) - J(\pi_\theta) \le \sum_t 2T\epsilon R_{max}$

# 3
## 3.1
python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Ant.pkl \
--env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
--expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
--video_log_freq -1 \
--no_gpu \
--train_batch_size 1000
## 3.2
python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Ant.pkl \
--env_name Ant-v4 --exp_name dagger_ant --n_iter 10 \
--do_dagger --expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
--video_log_freq -1 \
--no_gpu 