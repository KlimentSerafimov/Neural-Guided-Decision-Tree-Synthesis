//
// Created by Kliment Serafimov on 2019-02-24.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_FIRSTORDERLEARNING_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_FIRSTORDERLEARNING_H

#include "net.h"
#include "Data.h"
#include "Header.h"
#include "net_and_score.h"

ofstream tout("trace.out");

template<typename datatype>
class firstOrderLearning
{
public:

    firstOrderLearning()
    {

    }

    net step(net prev_step, double rate)
    {
        //prev_step.printWeights();
        net ret = net(prev_step);
        ret.perturb(rate);
        //ret.printWeights();
        return ret;
    }

    net deep_step(net prev_step, double rate) {
        int backtrack_potential = 3;

        int search_width = 3;

        // if search_width steps are is backtrack take best step; if backtrack do it only if: (1) if [ [sum_train_iter of # of candidates that are better than proposed step] OR absolute difference] < [backtrack potential]
        // and (2) second attempt from current learner;

        // track somehow #of itterations steps forward (have to take into account how many other threads reached that perticular hight) vs #itteration steps cost vs #iteration steps backwards (same as before)

        // track progress of roots via their kids.


        //need one big priority queue with stocastic swarm search with decaying tree like search that atributes credit to branches that have shown/are showing progress;



        //BUT ALSO JUST SEE WHAT ARE THE DELTAS!! DONE


        //ALSO DO DP ON THE STRUCTURE OF THE NEURAL NETWORK. NO NEED FOR ALL CONNECTED

        return net();
    }

    /*void progressive_train(int n)
    {
        net_and_score learner = net(n, 2*n, 1);

        int k_iter = 30;
        //performs meta training, using the max error after k iter of a task as signal.


        vector<datatype> local_data;

        for(int i = 0;i<data_of_functions.size();)
        {
            int data_step_size = 5;
            for(int k = 0; k < data_step_size && i<data_of_functions.size();i++, k++)
            {
                local_data.pb(data_of_functions[i]);
            }
            reptile_SA_train(learner, local_data, k_iter);
            evaluate_learner(learner, test_data_of_functions, true, 300);
        }

    }*/
/*
    void custom_distribution_reptile_step(
            net &try_learner,
            errorAsClassificatorData data,
            int iter_cutoff,
            double treshold)
    {
        //for(int i = 0;i<data.size();i++)
        int i;

        i = rand(0, data.size()-1);

        {
            net local_try_learner = net(try_learner.copy());
            net::parameters param = cutoff_param(iter_cutoff, 0.01);
            int num_iter = local_try_learner.train(&data[i], param, &net::softPriorityTrain);

            double local_error = local_try_learner.get_error_of_data(&data[i]);
            local_try_learner.minus(try_learner);
            local_try_learner.mul(-1.0/data.size());
            try_learner.minus(local_try_learner);
        }

    }
    
    void custom_distribution_meta_learn(
            net_and_score &global_best_solution,
            errorAsClassificatorData custom_data,
            int k_iter_init,
            double treshold)
    {

        int k_iter = k_iter_init;

        int global_stagnation = 0;

        net_and_score at_walking_solution;

        net_and_score best_solution;

        net_and_score SA_iter_best_solution = best_solution = at_walking_solution =
                evaluate_learner(global_best_solution, data, false, k_iter, treshold);

        for(int k_iter_loops = 0; global_stagnation < 4; k_iter--)
        {
            cout << "NEW k_iter = " << k_iter << endl;
            at_walking_solution = evaluate_learner(SA_iter_best_solution, data, false, k_iter, treshold);

            int count_stagnation = 0;

            assert(k_iter>0);
            for (int iter = 1; at_walking_solution.num_train_fail > 0 && global_stagnation < 4; iter++) {

                custom_distribution_reptile_step(at_walking_solution, data, k_iter, treshold);



                //at_walking_solution.printWeights();

                int repeat_const = 3;

                int repeat_count = (repeat_const*data.size());

                if (iter % repeat_count == 0) {

                    k_iter_loops+=repeat_count;
                    net_and_score new_score = at_walking_solution = evaluate_learner(at_walking_solution, data, true, k_iter, treshold);

                    //cout << new_score << endl;

                    //at_walking_solution.printWeights();


                    if (new_score < SA_iter_best_solution) {
                        SA_iter_best_solution = new_score;
                        count_stagnation = 0;
                        if(SA_iter_best_solution < best_solution)
                        {
                            global_stagnation = 0;
                            best_solution = SA_iter_best_solution;
                            if(best_solution < global_best_solution)
                            {
                                global_best_solution = best_solution;
                            }
                        }
                    }
                    else
                    {
                        count_stagnation++;
                        if(count_stagnation >= log2(data.size()))
                        {
                            net next_step = step(at_walking_solution, 16*SA_iter_best_solution.max_error/(data[0].size()*data.size()));
                            cout << "SA step" << endl;
                            new_score = SA_iter_best_solution = at_walking_solution = evaluate_learner(next_step, data, true, k_iter, treshold);

                            count_stagnation=0;
                            global_stagnation++;
                            k_iter = k_iter_init;

                            if(SA_iter_best_solution < best_solution)
                            {
                                global_stagnation = 0;
                                best_solution = SA_iter_best_solution;
                                if(best_solution < global_best_solution)
                                {
                                    global_best_solution = best_solution;
                                }
                            }
                        }
                    }

                    cout << "local stangation = " << count_stagnation <<endl;
                    cout << "global stangation = " << global_stagnation <<endl;
                    cout << "k_iter = " << k_iter << endl;
                    cout << "k_iter_loops = " << k_iter_loops << endl;
                    cout << iter << " at            = " << new_score.print() << endl;
                    cout << iter << " SA local best = " << SA_iter_best_solution.print() << endl;
                    cout << iter << " reptile  best = " << best_solution.print() << endl;

                    tout << new_score.clean_print() << endl;

                }
            }
        }
    }
*/

    /*void reptile_SA_train(
            net_and_score &global_best_solution,
            vector<datatype> data,
            int k_iter_init,
            double treshold)
    {
        k_iter_init = 60;

        int k_iter = k_iter_init;

        int global_stagnation = 0;

        net_and_score at_walking_solution;

        net_and_score best_solution;

        net_and_score SA_iter_best_solution = best_solution = at_walking_solution =
                evaluate_learner(global_best_solution, data, false, k_iter, treshold);

        for(int k_iter_loops = 0; global_stagnation < 4; k_iter--)
        {
            cout << "NEW k_iter = " << k_iter << endl;
            at_walking_solution = evaluate_learner(SA_iter_best_solution, data, false, k_iter, treshold);

            int count_stagnation = 0;

            assert(k_iter>0);
            for (int iter = 1; at_walking_solution.num_train_fail > 0 && global_stagnation < 4; iter++) {

                reptile_step(at_walking_solution, data, k_iter, treshold);

                //at_walking_solution.printWeights();

                int repeat_const = 3;

                int repeat_count = (repeat_const*data.size());

                if (iter % repeat_count == 0) {

                    k_iter_loops+=repeat_count;
                    net_and_score new_score = at_walking_solution = evaluate_learner(at_walking_solution, data, true, k_iter, treshold);

                    //cout << new_score << endl;

                    //at_walking_solution.printWeights();


                    if (new_score < SA_iter_best_solution) {
                        SA_iter_best_solution = new_score;
                        count_stagnation = 0;
                        if(SA_iter_best_solution < best_solution)
                        {
                            global_stagnation = 0;
                            best_solution = SA_iter_best_solution;
                            if(best_solution < global_best_solution)
                            {
                                global_best_solution = best_solution;
                            }
                        }
                    }
                    else
                    {
                        count_stagnation++;
                        if(count_stagnation >= log2(data.size()))
                        {
                            net next_step = step(at_walking_solution, 16*SA_iter_best_solution.max_error/(data[0].size()*data.size()));
                            cout << "SA step" << endl;
                            new_score = SA_iter_best_solution = at_walking_solution = evaluate_learner(next_step, data, true, k_iter, treshold);

                            count_stagnation=0;
                            global_stagnation++;
                            k_iter = k_iter_init;

                            if(SA_iter_best_solution < best_solution)
                            {
                                global_stagnation = 0;
                                best_solution = SA_iter_best_solution;
                                if(best_solution < global_best_solution)
                                {
                                    global_best_solution = best_solution;
                                }
                            }
                        }
                    }

                    cout << "local stangation = " << count_stagnation <<endl;
                    cout << "global stangation = " << global_stagnation <<endl;
                    cout << "k_iter = " << k_iter << endl;
                    cout << "k_iter_loops = " << k_iter_loops << endl;
                    cout << iter << " at            = " << new_score.print() << endl;
                    cout << iter << " SA local best = " << SA_iter_best_solution.print() << endl;
                    cout << iter << " reptile  best = " << best_solution.print() << endl;

                    tout << new_score.clean_print() << endl;

                }
            }
        }
    }*/


    void update_solutions(net_and_score &SA_iter_best_solution, net_and_score &best_solution,
                          net_and_score &global_best_solution,
                          int &global_stagnation, int &SA_stagnation)
    {
        if(SA_iter_best_solution < best_solution)
        {
            SA_stagnation = 0;

            best_solution.update(SA_iter_best_solution);

            if(best_solution < global_best_solution)
            {
                global_stagnation = 0;

                global_best_solution.update(best_solution);

            }
        }
    }

    void meta_learn(
            net_and_score &global_best_solution,
            vector<datatype> f_data,
            bool print,
            int leaf_iter_init,
            double treshold)
    {
        reptile_SA_train(global_best_solution, f_data, print, leaf_iter_init, treshold);
    }

    void order_tasks_by_difficulty_via_mutli_task_learning(
            net_and_score &global_best_solution,
            vector<datatype> f_data,
            bool print,
            int leaf_iter_init,
            double treshold)
    {
        assert(0);
    }

    class Policy
    {
    public:

        enum policyType : int {total_drift, controled_drift};

        policyType policy_type;

        int n;

        vector<DecisionTreeScore> desired;

        vector<double> observed;

        Policy()
        {

        }

        Policy(vector<Data> f_data)
        {
            policy_type = total_drift;
            n = f_data.size();
        }

        Policy(vector<DataAndScore> f_data)
        {
            policy_type = controled_drift;
            n = f_data.size();
            for(int i = 0;i<f_data.size();i++)
            {
                desired.pb(f_data[i].score);
            }
        }


        vector<int> id_to_consider;

        void update(net_and_score &generator)
        {
            if(policy_type == total_drift)
            {
                //no update
            }
            else if(policy_type == controled_drift)
            {
                id_to_consider.clear();

                observed = generator.individual_max_errors;

                assert(controled_drift_invariant());
                vector<pair<DecisionTreeScore, int> > desired_ordered;

                vector<pair<double, int> > observed_ordered;

                for (int i = 0; i < desired.size(); i++) {
                    desired_ordered.pb(mp(desired[i], i));
                    observed_ordered.pb(mp(observed[i], i));
                }

                sort_v(desired_ordered);
                sort_v(observed_ordered);

                bool mismatch = false;

                assert(desired_ordered.size() >= 1);
                DecisionTreeScore threshold_to_train_under = desired_ordered[0].f;

                for (int i = 0; i < desired_ordered.size(); i++) {
                    cout << desired_ordered[i].f.print() <<" " ;
                    if (!mismatch) {
                        if (desired_ordered[i].f < desired[observed_ordered[i].s]) {
                            mismatch = true;
                            threshold_to_train_under = desired[observed_ordered[i].s];
                        } else {
                            //id_to_consider.pb(observed_ordered[i].s);
                        }
                    } else {
                        if (desired[observed_ordered[i].s] < threshold_to_train_under) {
                            id_to_consider.pb(observed_ordered[i].s);
                        }
                    }
                }



                cout << endl;

                vector<DecisionTreeScore> for_bubble_sort;

                for(int i = 0;i<observed_ordered.size();i++) {
                    cout << desired[observed_ordered[i].s].print() << " ";

                    for_bubble_sort.pb(desired[observed_ordered[i].s]);
                }
                cout << endl;

                int count = 0;
                for(int i = 0; i<for_bubble_sort.size();i++)
                {
                    for(int j = for_bubble_sort.size()-2;j>=i;j--)
                    {
                        if(for_bubble_sort[j+1] < for_bubble_sort[j])
                        {
                            swap(for_bubble_sort[j], for_bubble_sort[j+1]);
                            count++;
                        }
                    }
                }

                cout << count <<endl;

                generator.ordering_error = count;

                cout << endl;

                cout << "threshold_to_train_under = " << threshold_to_train_under.print() <<endl;

                sort_v(id_to_consider);

                vector<int> tmp;

                tmp = id_to_consider;

                id_to_consider.clear();

                for(int i = 0;i<min(4, (int)tmp.size());i++)
                {
                    id_to_consider.pb(tmp[i]);
                }
            }
            else
            {
                assert(0);
            }
        }

        bool controled_drift_invariant()
        {
            return desired.size() == observed.size();
        }

        int query()
        {
            if(policy_type == total_drift)
            {
                return rand(0, desired.size()-1);
            }
            else if(policy_type == controled_drift)
            {
                if(id_to_consider.size() >= 1) {

                    int sample = rand(0, id_to_consider.size() - 1);
                    cout << desired[id_to_consider[sample]].print() << " ";
                    return id_to_consider[rand(0, id_to_consider.size() - 1)];
                }
                else
                {
                    return rand(0, desired.size()-1);
                }
            }
            else
            {
                assert(0);
            }
        }
    };

    void reptile_SA_train(
            net_and_score &global_best_solution,
            vector<datatype> f_data,
            bool print,
            int leaf_iter_init,
            double treshold)
    {
        double max_treshold = treshold;
        double min_treshold = (2*max_treshold)/3;

        int k_iter = leaf_iter_init;

        int global_stagnation = 0;

        Policy local_policy(f_data);
        local_policy.policy_type = Policy::total_drift;

        net_and_score at_walking_solution;

        net_and_score best_solution;

        net_and_score SA_iter_best_solution = best_solution = at_walking_solution =
                evaluate_learner(global_best_solution, f_data, false, k_iter, treshold);

        local_policy.update(at_walking_solution);

        if(best_solution < global_best_solution)
        {
            global_best_solution = best_solution;
        }

        bool enter = false;

        int give_up_after = 4;

        for(int k_iter_loops = 0; global_stagnation < give_up_after; )
        {


            if(enter)
            {
                local_policy.policy_type = Policy::controled_drift;
                treshold*=0.8;
                treshold = max(treshold, min_treshold);
                if(treshold == min_treshold)
                {
                    k_iter--;
                }
                at_walking_solution = evaluate_learner(SA_iter_best_solution, f_data, false, k_iter, treshold);
                local_policy.update(at_walking_solution);

            }
            if(print)
            {
                cout << "NEW k_iter = " << k_iter <<"; NEW treshold = " << treshold << endl;
            }

            enter = true;

            int count_stagnation = 0;

            int SA_stagnation = 0;

            assert(k_iter>0);

            for (int iter = 1; at_walking_solution.num_train_fail > 0 && global_stagnation < give_up_after; iter++) {

                reptile_step(at_walking_solution, f_data, k_iter, treshold, local_policy);

                //at_walking_solution.printWeights();

                double repeat_const;

                if(at_walking_solution.max_error < 0.45)
                {
                    repeat_const = 0.5;
                }
                else{

                    repeat_const = 2;
                }

                int repeat_count = 0+1*(repeat_const*f_data.size());

                if (iter % repeat_count == 0) {

                    cout << endl;

                    k_iter_loops+=repeat_count;
                    at_walking_solution = evaluate_learner(at_walking_solution, f_data, print, k_iter, treshold);

                    local_policy.update(at_walking_solution);


                    if(at_walking_solution.max_error < 0.45)
                    {

                        local_policy.policy_type = Policy::controled_drift;
                    }
                    else{

                        local_policy.policy_type = Policy::total_drift;
                    }

                    //cout << new_score << endl;

                    //at_walking_solution.printWeights();

                    if (at_walking_solution < SA_iter_best_solution) {
                        SA_iter_best_solution.update(at_walking_solution);
                        count_stagnation = 0;

                        update_solutions
                                (SA_iter_best_solution, best_solution, global_best_solution,
                                 global_stagnation, SA_stagnation);
                    }
                    else
                    {
                        count_stagnation++;
                        SA_stagnation++;



                        if(count_stagnation >= 2+log2(f_data.size()) || SA_stagnation >= 1+2*log2(f_data.size()))
                        {
                            double step_radius_size =
                                    16*SA_iter_best_solution.max_error/(f_data[0].size()*f_data.size());

                            net next_step = step(at_walking_solution, step_radius_size);
                            if(print)
                            {
                                cout << "SA step" << endl;
                            }
                            at_walking_solution = evaluate_learner(next_step, f_data, print, k_iter, treshold);


                            local_policy.update(at_walking_solution);

                            count_stagnation=0;
                            global_stagnation++;
                            k_iter = leaf_iter_init;

                            SA_iter_best_solution.update(at_walking_solution);

                            update_solutions
                                    (SA_iter_best_solution, best_solution, global_best_solution,
                                     global_stagnation, SA_stagnation);
                        }
                    }

                    if(true) {
                        cout << "local stangation = " << count_stagnation << endl;
                        cout << "global stangation = " << global_stagnation << endl;
                        cout << "k_iter = " << k_iter << endl;
                        cout << "k_iter_loops = " << k_iter_loops << endl;
                        cout << iter << " at            = " << at_walking_solution.print() << endl;
                        cout << iter << " SA local best = " << SA_iter_best_solution.print() << endl;
                        cout << iter << " reptile  best = " << best_solution.print() << endl;

                        tout << at_walking_solution.clean_print() << endl;
                    }
                }
            }
        }
    }




    void reptile_train(
            net &global_best_solution,
            vector<Data> data,
            int root_iter_cutoff,
            int k_iter_init,
            double treshold)
    {
        //cout << "in reptile_train" <<endl;
        for(int i = 0; i < root_iter_cutoff; i++)
        {
            reptile_step(global_best_solution, data, k_iter_init, treshold);
        }
    }

    void reptile_step(
            net &try_learner,
            vector<datatype> data,
            int iter_cutoff,
            double treshold,
            int i)
    {
        net local_try_learner = net(try_learner.copy());
        net::parameters param = cutoff_param(iter_cutoff, 0.01);
        int num_iter = local_try_learner.train(&data[i], param, &net::softPriorityTrain);

        double local_error = local_try_learner.get_error_of_data(&data[i]);
        local_try_learner.minus(try_learner);
        local_try_learner.mul(-0.33);
        try_learner.minus(local_try_learner);
    }

    void reptile_step(
            net &try_learner,
            vector<datatype> data,
            int iter_cutoff,
            double treshold)
    {
        int i = rand(0, data.size()-1);
        reptile_step(try_learner, data, iter_cutoff, treshold, i);
    }

    void reptile_step(
            net &try_learner,
            vector<datatype> data,
            int iter_cutoff,
            double treshold,
            Policy local_policy)
    {
        //for(int i = 0;i<data.size();i++)
        int i = local_policy.query();
        reptile_step(try_learner, data, iter_cutoff, treshold, i);
    }

    void evaluate_unit_task_print(DataAndScore* data, int local_iter, double local_error)
    {
        cout << data->print() <<"\t"<< local_iter << "\t" << local_error << endl;
    }

    void evaluate_unit_task_print(Data* data, int local_iter, double local_error)
    {
        cout << data->printConcatinateOutput() <<"\t"<< local_iter << "\t" << local_error << endl;
    }

    net_and_score evaluate_unit_task(net try_learner, datatype* data, bool print, int iter_cutoff, double treshold,
            net_and_score &score)
    {
        //cout << i <<endl;
        net local_try_learner = net(try_learner.copy());
        net::parameters param = cutoff_param(iter_cutoff, 0.01);
        int local_iter = local_try_learner.train(data, param, &net::softPriorityTrain);
        net_and_score individual_score;
        individual_score.is_init_score = false;
        score.is_init_score = false;

        double local_error = local_try_learner.get_error_of_data(data);

        score.max_error = max(score.max_error, local_error);
        score.sum_error += local_error;
        score.update_max_leaf_iter(local_iter);
        individual_score.max_error = local_error;
        individual_score.sum_error = local_error;

        if(local_iter == iter_cutoff && local_error > treshold)
        {
            score.num_train_fail++;
            individual_score.num_train_fail = 1;
            //break;
        }
        else
        {
            individual_score.max_train_iter = local_iter;
            score.max_train_iter = max(score.max_train_iter, local_iter);
            score.sum_train_iter+=local_iter;
        }

        if(print)
        {
            evaluate_unit_task_print(data, local_iter, local_error);
        }

        return individual_score;
    }

    net_and_score evaluate_learner(
            net try_learner,
            vector<datatype> data,
            bool print,
            int iter_cutoff,
            double treshold)
    {
        //cout << "in evaluate_learner" <<endl;

        vector<net_and_score> individual_scores;

        net_and_score score = try_learner;

        score.clear_vals();

        for(int i = 0;i<data.size();i++)
        {
            individual_scores.pb(evaluate_unit_task(try_learner, &data[i], print, iter_cutoff, treshold, score));
        }

        score.set_individual_scores(individual_scores);

        //cout << "average = " << (double)sum_train_iter/data_of_functions.size() <<endl;
        return score;
    }


    void meta_train(net_and_score learner, vector<datatype> data, double treshold)
    {

        int k_iter = 30;
        //performs meta training, using the max error after k iter of a task as signal.
        reptile_SA_train(learner, data, k_iter, treshold, true);
        evaluate_learner(learner, data, true, k_iter, treshold);
    }

    /*net train_on_local_family()
    {
        bool prev_print_close_local_data_model = print_close_local_data_model;
        print_close_local_data_model = false;
        int init_F_size = (int)data_of_functions.size();
        for(int i = 0;i<init_F_size;i++)
        {
            for(int j = 0; j<data_of_functions[i].numOutputs; j++)
            {
                for(int k = 0;k<data_of_functions[i].size();k++)
                {
                    datatype new_in_F = data_of_functions[i];
                    new_in_F.out[k][j].flip_value();
                    data_of_functions.push_back(new_in_F);
                }
            }
        }

        train_old();

        print_close_local_data_model = prev_print_close_local_data_model;

        return net(learner);
    }*/

    void train_old(net &the_learner, vector<datatype> data, double treshold)
    {
        net_and_score init_score = the_learner;
        init_score.is_init_score = true;
        train_SA(init_score, data, 800, treshold);
        the_learner = init_score;
    }

    bool pass(net_and_score prev_energy, net_and_score new_energy, double temperature, net_and_score best)
    {
        assert(prev_energy.has_value());
        if(!new_energy.has_value())//failed test
        {
            cout << "fail" <<endl;
            return false;
        }
        //cout << "compare = " << new_energy <<" "<< prev_energy << endl;
        if(new_energy < prev_energy)
        {
            cout << "take" <<endl;
            return true;
        }
        else
        {
            double acc = 0.15*temperature*(prev_energy*0.8+best*0.2)/new_energy;//exp(-(new_energy/prev_energy)/(temperature));
            //cout << "e, e', acc, temp = " << prev_energy << " "<< new_energy <<" " << acc << " " << temperature << endl;
            bool here =  acc > (double)rand(0, 1000)/1000;
            //assert(!here);
            //cout << "here = " << here <<endl;
            return here;
        }
    }

    void train_SA(net &best_soution, vector<datatype> data, int max_iter, double treshold)
    {

        int num_SA_steps = 40;
        net_and_score at = best_soution;
        net_and_score prev_solution;
        do
        {
            cout << "SA WITH ITER = " << num_SA_steps << endl;
            prev_solution = at;
            //at = best_soution;
            double init_diameter = 0.6;
            double min_diameter = 0.001;
            simulated_annealing(at, data, max_iter, num_SA_steps, init_diameter, min_diameter, 0, treshold);
            num_SA_steps*=2;
            cout << "compare = " << at.print() <<" "<< prev_solution.print() <<endl;
        }while(at < prev_solution && num_SA_steps <= 160);
        best_soution = prev_solution;
    }

    void simulated_annealing(
                net_and_score &best_solution,
                vector<datatype> data,
                int max_iter,
                int num_SA_steps,
                double init_diameter,
                double min_diameter,
                int depth_left,
                double treshold)
    {
        best_solution = evaluate_learner(best_solution, data, false, 1000, treshold);

        double at_diameter = init_diameter;
        net_and_score at_walking_solution = best_solution;

        for(int iter = 1; iter < num_SA_steps; iter++)
        {
            at_diameter = init_diameter - iter*(init_diameter-min_diameter)/num_SA_steps;

            if(depth_left == 1)cout << "at_D = " << at_diameter <<endl;

            net next_step = step(at_walking_solution, at_diameter);

            //int old_iter_cutoff = at_walking_solution.max_train_iter*2+(!at_walking_solution.has_value())*treshold;


            net_and_score next_step_evaluation =
                    evaluate_learner(next_step, data, false, max_iter, treshold);

            //cout << "consider = " << next_step_evaluation << endl;

            if(false && depth_left)
            {
                assert(0);
                simulated_annealing(next_step_evaluation, data, max_iter, 6, at_diameter/2, min_diameter/10, depth_left-1, treshold);
                next_step = next_step_evaluation;
                next_step_evaluation = evaluate_learner(next_step, data, false, max_iter, treshold);
            }

            //next_step_evaluation = local_entropy_learner_evaluation(local_local_learner, data, false, 800);

            if(pass(at_walking_solution, next_step_evaluation, (double)(num_SA_steps-iter)/num_SA_steps, best_solution))
            {
                //cout << "HERE" <<endl;
                at_walking_solution = next_step_evaluation;
                //cout << "at_Walking solution = " << at_walking_solution <<endl;
                if(at_walking_solution<best_solution)
                {
                    //cout << "ULTIMATE HERE" <<endl;
                    best_solution = at_walking_solution;

                    //local_learner.printWeights();
                }
            }
            else
            {

            }
            if(depth_left == 1)
            {
                cout << iter << " at   =  " << at_walking_solution.print() <<endl;
                cout << iter << " best = " << best_solution.print() <<endl;
            }
            /*if(best_solution.max_train_iter <= 1)
            {
                break;
            }*/
        }
    }

    /*void expand_iterative_data(vector<datatype> &iterative_data, int batch_size)
    {
        for(int i = (int)iterative_data.size(), init_i = i; iterative_data.size()<init_i+batch_size && i < data_of_functions.size(); i++)
        {
            iterative_data.push_back(data_of_functions[i]);
        }
    }*/

    bool learner_treshold_test_on_iterative_sapce(net learner, vector<datatype> iterative_data, int max_cutoff, double treshold)
    {
        net_and_score num_iter = evaluate_learner(learner, iterative_data, false, max_cutoff, treshold);
        return num_iter.has_value();
    }

    /*void progressive_train()
    {
        learner = net(n, n, 1);
        //learner.set_special_weights();

        vector<datatype> iterative_data;

        int threshold = 1;
        for(int i = 0, batch_size = 1; i<data_of_functions.size(); i += batch_size)
        {
            bool succeed = false;
            vector<datatype> local_iterative_data;

            while(!succeed)
            {
                local_iterative_data.clear();
                vector<net_and_score> individual_scores;
                evaluate_learner(learner, data_of_functions, false, threshold, individual_scores);
                for(int i = 0;i<individual_scores.size();i++)
                {
                    if(individual_scores[i].has_value())
                    {
                        local_iterative_data.push_back(data_of_functions[i]);
                    }
                }
                //succeed = (local_iterative_data.size() > 2*iterative_data.size() || local_iterative_data.size() == data_of_functions.size());
                succeed = (local_iterative_data.size() > iterative_data.size());
                threshold*=2;
            }



            batch_size = local_iterative_data.size() - iterative_data.size();
            iterative_data = local_iterative_data;
            double rez = evaluate_learner(learner, local_iterative_data, true);
            cout << "max_train_iter = " << rez <<endl;
            double train_threshold = max(300.0, rez);

            //cout << "train threshold = " << train_threshold << endl;

            train(learner, iterative_data, train_threshold);

            threshold = (evaluate_learner(learner, iterative_data, false, learner.max_train_iter+1).max_train_iter+1)*2.5;
            assert(threshold!=-1 && threshold >= 2);
            cout << "i, iterative_data_size = " << i+batch_size <<" "<< iterative_data.size() <<endl;
        }

    }
    */

    int local_entropy_learner_evaluation(net try_learner, vector<datatype> data, bool print, int iter_cutoff, double treshold)
    {
        int c = 10;
        int rez = 0;
        bool fail = false;
        for(int i = 0; i<c; i++)
        {
            net local_step_learner = step(try_learner, 0.05);
            int local_rez =  evaluate_learner(local_step_learner, data, false, 800, treshold);
            rez = max(rez, local_rez);
            if(local_rez == -1)
            {
                fail = true;
            }
            cout << local_rez <<endl;
        }
        if(fail)
        {
            return -1;
        }
        return rez;
    }
};

#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_FIRSTORDERLEARNING_H
