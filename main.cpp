
/* Code written by Kliment Serafimov */

#include "neuron.h"
#include "layer.h"
#include "Data.h"
#include "Header.h"
#include "bit_signature.h"
#include "util.h"
#include "layered_graph.h"
#include "batch.h"
#include "net.h"
#include "decision_tree.h"
#include "firstOrderLearning.h"

ofstream fout("table.out");

template<typename datatype>
class dp_decision_tree
{
public:
    struct dp_ret
    {
        int w;
        int w_root;

        int h;
        int h_root;

        int num_solutions;

        void init()
        {
            w = (1<<30);
            h = (1<<30);
            num_solutions = 0;
        }

        void update(dp_ret left, dp_ret right, int mask_used)
        {
            if(left.w + right.w + 1 < w)
            {
                w = left.w + right.w+1;
                w_root = mask_used;


                h = max(left.h, right.h)+1;
                h_root = mask_used;

                num_solutions = left.num_solutions*right.num_solutions;
                assert(num_solutions < (1<<15));
            }
            else if(left.w + right.w + 1 == w)
            {
                num_solutions += left.num_solutions*right.num_solutions;
                assert(num_solutions < (1<<15));
            }
            /*if(max(left.h, right.h)+1 < h)
            {
                h = max(left.h, right.h)+1;
                h_root = mask_used;
            }*/
        }

        void update(
                int mask,
                int *dp_w,
                int *dp_w_root,

                int *dp_h,
                int *dp_h_root,

                int *dp_num_solutions
        )
        {
            if(w<dp_w[mask])
            {
                dp_w[mask] = w;
                dp_w_root[mask] = w_root;

                dp_h[mask] = h;
                dp_h_root[mask] = h_root;

                dp_num_solutions[mask] = num_solutions;
            }
            /*if(h<dp_h[mask])
            {
                dp_h[mask] = h;
                dp_h_root[mask] = h_root;
            }*/
        }
        string print()
        {
            string ret = "w = " + to_string(w) + " h = "+ to_string(h);
            return ret;
        }
    };

    bool vis[(1<<16)];

    int dp_w[(1<<16)];
    int dp_w_root[(1<<16)];

    int dp_h[(1<<16)];
    int dp_h_root[(1<<16)];

    int dp_num_solutions[(1<<16)];

    dp_ret dp_data[1<<16];


    dp_ret rek(int n, datatype *the_data, int mask)
    {
        assert(mask < (1<<16));
        if(!vis[mask])
        {
            vis[mask] = true;
            if(the_data->is_constant())
            {
                bit_signature out  = 0;
                for(int i = 0;i<the_data->size();i++)
                {
                    if(i == 0)
                        out = the_data->out[i][0];
                    else
                    {
                        assert(out == the_data->out[i][0]);
                    }
                }
                dp_w[mask] = dp_h[mask] = 1;
                dp_w_root[mask] = dp_w_root[mask] = mask;
                dp_num_solutions[mask] = 1;
            }
            else
            {
                vector<int> active_nodes = the_data->get_active_bits();

                dp_ret ret;
                ret.init();

                for(int i = 0;i<active_nodes.size();i++)
                {
                    datatype left, right;
                    bit_signature split_idx = bit_signature(active_nodes[i]);

                    assert(split_idx.vector_id == active_nodes[i]);
                    the_data->split(split_idx, left, right);

                    int left_mask = 0;
                    int right_mask = 0;

                    for(int j = 0, in_row = 0;j<(1<<n);j++)
                    {
                        if((mask&(1<<j)) != 0)
                        {
                            assert(in_row < the_data->in.size());
                            assert(split_idx.vector_id < the_data->in[in_row].size());

                            if(the_data->in[in_row][split_idx.vector_id] == 1)
                            {
                                left_mask|=(1<<j);
                            }
                            else if(the_data->in[in_row][split_idx.vector_id] == -1)
                            {
                                right_mask|=(1<<j);
                            }
                            in_row++;
                        }
                    }

                    assert((left_mask|right_mask) == mask);
                    //cout << bitset<16>(left_mask).to_string() <<endl;
                    //cout << bitset<16>(right_mask).to_string() <<endl;
                    //cout << endl;

                    dp_ret left_ret = rek(n, &left, left_mask);
                    dp_ret right_ret = rek(n, &right, right_mask);

                    ret.update(left_ret, right_ret, left_mask);

                }

                /*vector<pair<int, int> > active_pairs;
                for(int i = 0;i<active_nodes.size();i++)
                {
                    for(int j = i+1;j<active_nodes.size(); j++)
                    {
                        active_pairs.push_back(mp(active_nodes[i], active_nodes[j]));
                    }
                }

                vector<operator_signature> operators = the_data->get_operators(active_pairs);
                */
                ret.update(mask, dp_w, dp_w_root, dp_h, dp_h_root, dp_num_solutions);
            }
        }

        return dp_ret{dp_w[mask], dp_w_root[mask], dp_h[mask], dp_h_root[mask], dp_num_solutions[mask]};
    }

    void print_tree(int n, int mask, int t)
    {
        if(n == 4)
        {
            cout << indent(t) << bitset<16>(mask).to_string() <<endl;
        }
        else if(n == 3)
        {
            cout << indent(t) << bitset<8>(mask).to_string() <<endl;
        }
        else if(n == 2)
        {

            cout << indent(t) << bitset<4>(mask).to_string() <<endl;
        }
        else
        {
            assert(0);
        }

        if(1 == dp_w[mask]) return;
        print_tree(n, dp_w_root[mask], t+1);
        print_tree(n, mask-dp_w_root[mask], t+1);
    }

    int rez[33];
    int num_rez[33];
    int num_correct[33];
    int num_wrong[33][11];

    //6.100
    //6.890
    //6.s081
    //6.033
    //6.UAT

    //6.s898
    //CMS.333

    //IOI with Vlade, Physics with Martin, intro to programming with Josif
    //IDEAS global challenge
    //Video for Fond for innovation
    

    DecisionTreeScore synthesize_decision_tree_and_get_size
        (net::parameters param, int n, datatype the_data, DecisionTreeSynthesiserType synthesizer_type)
    {
        DecisionTreeScore ret = DecisionTreeScore(synthesizer_type);

        if(synthesizer_type == optimal)
        {

            memset(vis, 0, sizeof(vis));
            memset(dp_w, 63, sizeof(dp_w));
            memset(dp_h, 63, sizeof(dp_h));

            if(n <= 4)
            {
                dp_ret opt;

                opt = rek(n, &the_data, (1<<(1<<n))-1);

                ret.size = opt.w;
                ret.num_solutions = opt.num_solutions;
            }
            else
            {
                assert(0);
            }

        }
        else
        {
            param.decision_tree_synthesiser_type = synthesizer_type;
            decision_tree confusion_extracted_tree = decision_tree(&the_data, param);
            ret.size = confusion_extracted_tree.size;
        }
        return ret;
    }


    /*DecisionTreeScore old_extract_decision_tree_and_compare_with_opt_and_entropy(net::parameters param, int n, Data the_data)
    
    {
        param.decision_tree_synthesiser_type = confusion_guided;
        decision_tree confusion_extracted_tree = decision_tree(&the_data, param);
        int confusion_guided_size = confusion_extracted_tree.size;

        param.decision_tree_synthesiser_type = entropy_guided;
        decision_tree entropy_based_tree = decision_tree(&the_data, param);
        int entropy_guided_size = entropy_based_tree.size;

        DecisionTreeScore ret;
        dp_ret opt;

        if(n <= 4)
        {
            opt = rek(n, &the_data, (1<<(1<<n))-1);
            ret.set_opt(opt.w);
        }

        ret.neural_guided_size = confusion_guided_size;
        ret.entropy_guided_size = entropy_guided_size;

        return ret;
    }

    DecisionTreeScore new_extract_decision_tree_and_compare_with_opt_and_entropy(net::parameters param, int n, Data the_data)
    {
        memset(vis, 0, sizeof(vis));
        memset(dp_w, 63, sizeof(dp_w));
        memset(dp_h, 63, sizeof(dp_h));

        param.decision_tree_synthesiser_type = neural_guided;
        decision_tree neurally_extracted_tree = decision_tree(&the_data, param);
        int neural_guided_size = neurally_extracted_tree.size;

        param.decision_tree_synthesiser_type = entropy_guided;
        decision_tree entropy_based_tree = decision_tree(&the_data, param);
        int entropy_guided_size = entropy_based_tree.size;

        DecisionTreeScore ret;
        dp_ret opt;

        if(n <= 4)
        {
            opt = rek(n, &the_data, (1<<(1<<n))-1);
            ret.set_opt(opt.w);
        }
        ret.neural_guided_size = neural_guided_size;
        ret.entropy_guided_size = entropy_guided_size;

        return ret;
    }*/

    void dp_init(int n, datatype the_data)
    {
        memset(vis, 0, sizeof(vis));
        memset(dp_w, 63, sizeof(dp_w));
        memset(dp_h, 63, sizeof(dp_h));
        dp_ret ret = rek(n, &the_data, (1<<(1<<n))-1);

        //the_data.printData("data");

        //cout << "tree" <<endl;
        //print_tree((1<<16)-1, 0);

        //cout <<"score = " << ret.print() <<endl;

        int id_iter_count = 8;//8;
        int id_ensamble_size = 10;

        net::parameters param = net::parameters(1, 1);
        param.set_number_resets(1);
        param.num_stale_iterations = 10000;
        param.set_iteration_count(id_iter_count);
        param.ensamble_size = id_ensamble_size;
        param.priority_train_f = &net::softPriorityTrain;//&net::harderPriorityTrain;
        param.first_layer = 2*n;
        param.ensamble_size = 1;

        //try to pick dimension that maximizes confusion differnece between the two segments;

        int heuristic_ret = 100;
        decision_tree best_tree = decision_tree();
        for(int i = 1;i<=1;i++)
        {
            decision_tree tree = decision_tree(&the_data, param);
            if(tree.size<heuristic_ret)
            {
                best_tree = tree;
                heuristic_ret = tree.size;
            }
            //heuristic_ret = min(heuristic_ret, tree.size);
            //cout << tree.size <<" "<< tree.height <<endl;
        }
        if(false || print_close_local_data_model)
        {
            if(heuristic_ret-ret.w >= 2)
            {
                the_data.printData(to_string(heuristic_ret-ret.w)+" errors");
                print_tree(n, (1<<(1<<n))-1, 0);
                best_tree.print_gate(0);
            }
            if(false && ret.w == 11 && heuristic_ret-ret.w == 0)
            {
                the_data.printData("0 errors");
                print_tree(n, (1<<(1<<n))-1, 0);

                best_tree.print_gate(0);
            }
        }
        //cout << "end" <<endl;
        rez[ret.w]+=heuristic_ret;
        num_rez[ret.w]++;
        num_correct[ret.w]+=(heuristic_ret == ret.w);
        num_wrong[ret.w][heuristic_ret-ret.w]++;
        cout << ret.w <<" "<< (double)rez[ret.w]/num_rez[ret.w] <<endl;
    }

    void run()
    {
        memset(rez, 0, sizeof(rez));
        memset(num_rez, 0, sizeof(num_rez));
        memset(num_correct, 0, sizeof(num_correct));
        memset(num_wrong, 0, sizeof(num_wrong));

        int unit_calc = 29;//495;
        vector<int> unit_calcs;
        //unit_calcs.push_back((1<<8)-1);
        //unit_calcs.push_back((1<<16)-1-(1<<4)-(1<<8));
        unit_calcs.push_back(unit_calc);
        unit_calcs.push_back(unit_calc);
        unit_calcs.push_back(unit_calc);
        unit_calcs.push_back(unit_calc);
        /*for(int i = 0;i<16;i++)
        {
            unit_calcs.push_back(unit_calc^(1<<i));
        }*/

        int n = 3;

        for(int i = 0;i<1000*0+1*(1<<(1<<n));i+=1+0*rand(0, 120))
            //for(int i = unit_calc;i<=unit_calc;i+=1+0*rand(0, 120))
            //for(int at = 0, i = unit_calcs[at]; at<unit_calcs.size();at++,  i = unit_calcs[at])
        {
            datatype new_data;

            new_data.init_exaustive_table_with_unary_output(n, i);
            //new_data.printData("init");
            cout << i<<" ";
            dp_init(n, new_data);
            /*
            if(i%100 == 0)
            {
                print_score();
            }*/
        }
        print_score();
    }

    void print_score()
    {
        for(int i = 1;i<31;i+=2)
        {
            cout << i <<"\t"<< (double)rez[i]/num_rez[i]  << "\t" << ((double)rez[i]/num_rez[i])/i << endl;
        }
        int sum_train_iter_correct = 0;
        int sum_train_iter = 0;
        for(int i = 1;i<31;i+=2)
        {
            cout << i <<"\t"<<"::"<<"\t";
            int local_correct = 0;
            int local_sum_train_iter = 0;
            for(int j = 0;j<=12;j+=2)
            {
                int num = num_wrong[i][j];
                if(j == 0)
                {
                    sum_train_iter_correct+=num;
                    local_correct+=num;
                }
                sum_train_iter+=num;
                local_sum_train_iter+=num;

                if(num == 0)
                {
                    cout << ".\t";
                }
                else
                {
                    cout << num <<"\t";
                }
            }
            cout << "correct :: \t" << local_correct << "\t" << local_sum_train_iter;
            cout << endl;
        }
        cout << "correct = " << sum_train_iter_correct <<"/"<<sum_train_iter <<endl;
    }
};

template<typename datatype>
class firstOrderDatasets
{
public:
    int n;
    vector<datatype> train_data;
    vector<datatype> test_data;
    firstOrderDatasets()
    {

    }

    void print()
    {
        cout << "TRAIN" <<endl;
        for(int i = 0;i<train_data.size();i++)
        {
            cout << train_data[i].print() << endl;
        }
        cout << "TEST" <<endl;
        for(int i = 0;i<test_data.size();i++)
        {
            cout << test_data[i].print() << endl;
        }

    }
};

template<typename datatype>
class secondOrderDatasets
{
public:
    int n;
    vector<vector<datatype> > train_meta_data;
    vector<vector<datatype> > test_meta_data;


};

class GeneralizeDataset
{
    Data train_data;
    Data generalize_data;//like validation set, but used in training
};

class firstOrderLearnToGeneralize
{
    vector<GeneralizeDataset> learn_to_generalize_data;
    vector<GeneralizeDataset> test_generalization_data;
};

template<typename datatype>
class secondOrderLearning: public firstOrderLearning<datatype>
{
public:

    secondOrderLearning(): firstOrderLearning<datatype>()
    {

    }

    void learn_to_meta_learn(
            meta_net_and_score &learner,
            vector<vector<Data> > train_meta_data,
            int root_iter, int leaf_iter, double treshold)
    {
        learning_to_reptile(learner, train_meta_data, root_iter, leaf_iter, treshold);
    }

    void learning_to_reptile(
            meta_net_and_score &global_best_solution,
            vector<vector<Data> > f_data,
            int root_iter_init,
            int leaf_iter_init,
            double treshold)
    {
        double max_treshold = treshold;
        double min_treshold = max_treshold/3;

        int root_iter = root_iter_init;
        int k_iter = leaf_iter_init;

        int global_stagnation = 0;

        meta_net_and_score at_walking_solution;

        meta_net_and_score best_solution;

        meta_net_and_score SA_iter_best_solution = best_solution = at_walking_solution =
                evaluate_meta_learner(global_best_solution, f_data, false, root_iter, k_iter, treshold);

        if(best_solution < global_best_solution)
        {
            global_best_solution = best_solution;
        }

        bool enter = false;

        int give_up_after = 1;

        for(int k_iter_loops = 0; global_stagnation < give_up_after; )
        {

            if(enter)
            {
                treshold*=0.8;
                treshold = max(treshold, min_treshold);
                if(treshold == min_treshold)
                {
                    k_iter--;
                }
                at_walking_solution = evaluate_meta_learner(SA_iter_best_solution, f_data, false, root_iter, k_iter, treshold);
            }
            cout << "NEW k_iter = " << k_iter <<"; NEW treshold = " << treshold << endl;

            enter = true;

            int count_stagnation = 0;

            int SA_stagnation = 0;

            assert(k_iter>0);

            for (int iter = 1; at_walking_solution.num_train_fail > 0 && global_stagnation < give_up_after; iter++) {

                meta_reptile_step(at_walking_solution, f_data, root_iter, k_iter, treshold);

                //at_walking_solution.printWeights();

                int repeat_const = 2;

                int repeat_count = 1+0*(repeat_const*f_data.size());

                if (iter % repeat_count == 0) {

                    k_iter_loops+=repeat_count;
                    net_and_score new_score = at_walking_solution =
                                                      evaluate_meta_learner(at_walking_solution, f_data, true, root_iter, k_iter, treshold);

                    //cout << new_score << endl;

                    //at_walking_solution.printWeights();

                    if (new_score < SA_iter_best_solution) {
                        SA_iter_best_solution = new_score;
                        count_stagnation = 0;

                        firstOrderLearning<datatype>::update_solutions
                                (SA_iter_best_solution, best_solution, global_best_solution,
                                 global_stagnation, SA_stagnation);
                    }
                    else
                    {
                        count_stagnation++;
                        SA_stagnation++;
                        if(count_stagnation >= 1+log2(f_data.size()) || SA_stagnation >= 1+2*log2(f_data.size()))
                        {

                            double radius = 16*SA_iter_best_solution.max_error/(f_data[0][0].size()*f_data[0].size()*f_data.size());
                            net next_step = firstOrderLearning<datatype>::step(at_walking_solution, radius);
                            cout << "SA step" << endl;
                            new_score = SA_iter_best_solution = at_walking_solution =
                                    evaluate_meta_learner(next_step, f_data, true, root_iter, k_iter, treshold);

                            count_stagnation=0;
                            global_stagnation++;
                            k_iter = leaf_iter_init;


                            firstOrderLearning<datatype>::update_solutions
                                    (SA_iter_best_solution, best_solution, global_best_solution,
                                     global_stagnation, SA_stagnation);
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

    void meta_reptile_step(
            net &try_learner,
            vector<vector<Data> > data,
            int root_cutoff,
            int leaf_cutoff,
            double treshold)
    {
        int i;

        i = rand(0, data.size()-1);

        {
            //cout << "train on task: " << i <<endl;
            //try_learner.printWeights();
            net local_try_learner = net(try_learner.copy());
            //net_and_score before_train = evaluate_learner(local_try_learner, data[i], print, leaf_cutoff);

            firstOrderLearning<datatype>::reptile_train(local_try_learner, data[i], root_cutoff, leaf_cutoff, treshold);

            local_try_learner.minus(try_learner);
            local_try_learner.mul(-1.0/data.size());
            try_learner.minus(local_try_learner);
        }

        //cout << "average = " << (double)sum_train_iter/Dataset_of_functions.size() <<endl;
    }

    meta_net_and_score evaluate_meta_learner(
            net try_learner,
            vector<vector<Data> > data,
            bool print,
            int root_iter,
            int leaf_iter,
            double treshold)
    {
        meta_net_and_score score = net_and_score(try_learner);

        score.clear_vals();

        if(print)
        {
            cout << "IN evaluate_meta_learner" << endl;
        }
        for(int i = 0;i<data.size();i++)
        {
            //cout << i <<endl;
            net local_try_learner = net(try_learner.copy());


            firstOrderLearning<datatype>::reptile_train(local_try_learner, data[i], root_iter, leaf_iter, treshold);

            net_and_score reptile_score =
                    firstOrderLearning<datatype>::evaluate_learner(local_try_learner, data[i], print, leaf_iter, treshold);

            score.is_init_score = false;
            score.max_error = max(score.max_error, reptile_score.max_error);
            score.sum_error += reptile_score.sum_error;
            score.num_train_fail += reptile_score.num_train_fail;
            score.max_leaf_iter = max(score.max_leaf_iter, reptile_score.max_leaf_iter);

            if(print)
            {
                cout << "\t" << i << "\t" << reptile_score.max_error << endl;
                //assert(0);//do the comment
                //tmp.pring_delta_w(try_learner);
                //local_try_learner.printWeights();
            }
        }
        if(print) {
            cout << "END evaluate_meta_learner" << endl;
        }
        //cout << "average = " << (double)sum_train_iter/Dataset_of_functions.size() <<endl;
        return score;
    }
};

class f_and_score: public DecisionTreeScore
{
public:
    long long f;

    f_and_score(int _f, DecisionTreeScore tmp): DecisionTreeScore()
    {
        f = _f;
        size = tmp.size;
        num_solutions = tmp.num_solutions;
    }

    operator int()
    {
        return f;
    }
};

DecisionTreeScore get_opt_size_of_decision_tree(int n, int f)
{
    Data new_data;
    new_data.init_exaustive_table_with_unary_output(n, f);

    dp_decision_tree<Data> decision_tree_solver;
    DecisionTreeScore size_opt;
    net::parameters cutoff_parametes;
    size_opt = decision_tree_solver.synthesize_decision_tree_and_get_size
            (cutoff_parametes, n, new_data, optimal);

    return size_opt;
}

vector<f_and_score> get_smallest_f(int n)
{

    assert(n <= 3);

    vector<int> fs;

    for (int i = 0; i < (1 << (1 << n)); i++) {
        fs.pb(i);
    }

    vector<f_and_score> ordering;

    for(int i = 0;i<fs.size();i++)
    {
        ordering.pb(f_and_score(fs[i], get_opt_size_of_decision_tree(n, fs[i])));
    }

    sort_v(ordering);

    return ordering;

}

firstOrderDatasets<DataAndScore> init_random_train_set(int n)
{
    firstOrderDatasets<DataAndScore> ret;
    ret.n = n;
    //train
    vector<int> function_ids;

    int test_num_samples = 256;

    vector<f_and_score> sample_ids;

    test_num_samples = 200;
    for(int i = 0; i < max(test_num_samples, test_num_samples);i++)
    {
        long long first = (long long)rand(0, (1<<31)) << 31;
        long long second = rand(0, (1 << 31));

        long long sample_id = (first + second);

        //cout << first << " " << second << " "<< sample_id << endl;
        sample_ids.pb(f_and_score(sample_id, get_opt_size_of_decision_tree(n, sample_id)));
    }

    //test

    for(int i = 0;i<test_num_samples;i++)
    {
        DataAndScore new_data(sample_ids[i]);
        new_data.init_exaustive_table_with_unary_output(n, sample_ids[i].f);
        ret.test_data.pb(new_data);
    }

    ret.print();

    return ret;
}

firstOrderDatasets<DataAndScore> init_custom_data_and_difficulty_set(int n)
{
    assert(n<=3);
    firstOrderDatasets<DataAndScore> ret;
    ret.n = n;
    //train
    vector<int> function_ids ;

    //int train_num_samples; //defined later
    int test_num_samples = (1<<(1<<n));


    vector<f_and_score> sample_ids = get_smallest_f(n);

    //train
    for (int i = 0; i < sample_ids.size(); i++)
    {
        if(sample_ids[i].size <= 5)
        {
            DataAndScore new_data(sample_ids[i]);
            new_data.init_exaustive_table_with_unary_output(n, sample_ids[i].f);
            ret.train_data.pb(new_data);
        }
    }

    sample_ids.clear();

    sample_ids = get_smallest_f(n);

    //test

    for(int i = 0;i<test_num_samples;i++)
    {
        DataAndScore new_data(sample_ids[i]);
        new_data.init_exaustive_table_with_unary_output(n, sample_ids[i].f);
        ret.test_data.pb(new_data);
    }

    ret.print();

    return ret;
}



/*
firstOrderDatasets<DataAndScore> init_smallest_train_set(int n)
{
    firstOrderDatasets<DataAndScore> ret;
    ret.n = n;
    //train
    vector<int> function_ids ;

    int train_num_samples = 256;
    int test_num_samples = (1<<(1<<n));

    vector<f_and_score> sample_ids = get_smallest_f(n, train_num_samples);

    //sample_ids.pb(0);
    //sample_ids.pb((1<<(1<<n))-1);

    //train
    for (int i = 0; i < train_num_samples; i++)
    {
        DataAndScore new_data(sample_ids[i].size, sample_ids[i].num_solutions);
        new_data.init_exaustive_table_with_unary_output(n, sample_ids[i].f);
        ret.train_data.pb(new_data);
    }

    sample_ids.clear();

    if(n<=4)
    {
        sample_ids = get_smallest_f(n, test_num_samples);
    }
    else if(n<=6)
    {
        test_num_samples = 200;
        for(int i = 0; i < max(train_num_samples, test_num_samples);i++)
        {
            long long first = (long long)rand(0, (1<<31)) << 31;
            long long second = rand(0, (1 << 31));

            long long sample_id = (first + second);

            //cout << first << " " << second << " "<< sample_id << endl;
            sample_ids.pb(f_and_score(sample_id, get_opt_size_of_decision_tree(n, sample_id)));
        }
    }
    else
    {
        assert(0);
    }

    //test

    for(int i = 0;i<test_num_samples;i++)
    {
        DataAndScore new_data(sample_ids[i].size, sample_ids[i].num_solutions);
        new_data.init_exaustive_table_with_unary_output(n, sample_ids[i].f);
        ret.test_data.pb(new_data);
    }

    ret.print();

    return ret;
}
*/

void init_exaustive_table_with_unary_output(Data &f_data, int n, int f_id)
{
    f_data.init_exaustive_table_with_unary_output(n, f_id);
}

void init_exaustive_table_with_unary_output(DataAndScore &f_data, int n, int f_id)
{
    f_data.init_exaustive_table_with_unary_output(n, f_id);

    dp_decision_tree<Data> decision_tree_solver;
    net::parameters cutoff_parametes;
    DecisionTreeScore size_opt = decision_tree_solver.synthesize_decision_tree_and_get_size
        (cutoff_parametes, n, f_data, optimal);

    f_data.score = size_opt;
}



template<typename datatype>
firstOrderDatasets<datatype> initFirstOrderDatasets(int n, int train_num_samples, int test_num_samples)
{
    firstOrderDatasets<datatype> ret;
    ret.n = n;
    //train
    vector<int> function_ids ;

   //int train_num_samples = 10;
   //int test_num_samples = 60;

    vector<long long> sample_ids;

    //common
    if(n<=4) {

        for (int i = 0; i < (1 << (1 << n)); i++) {
            sample_ids.pb(i);
        }

        //shuffle(sample_ids.begin(), sample_ids.end(), std::default_random_engine(0));

    }
    else if(n<=6)
    {
        for(int i = 0; i < test_num_samples;i++)
        {
            long long first = (long long)rand(0, (1<<31)) << 31;
            long long second = rand(0, (1 << 31));

            long long sample_id = (first + second);

            //cout << first << " " << second << " "<< sample_id << endl;
            sample_ids.pb(sample_id);
        }
    }
    else
    {
        assert(0);
    }

    //train
    for (int i = 0; i < train_num_samples; i++) {

        datatype next_f;

        init_exaustive_table_with_unary_output(next_f, n, sample_ids[i]);

        ret.train_data.pb(next_f);
    }

    //test

    for(int i = 0;i<test_num_samples;i++)
    {
        datatype next_f;

        init_exaustive_table_with_unary_output(next_f, n, sample_ids[i]);

        ret.test_data.pb(next_f);
    }

    return ret;
}

template<typename datatype>
firstOrderDatasets<datatype> completeFirstOrderTestDataset(int n)
{
    return initFirstOrderDatasets<datatype>(n, 0, (1<<(1<<n)));
}

vector<vector<Data> > initSecondOrderDataset(vector<Data> data, int subDataset_size)
{
    vector<vector<Data> > ret;
    for(int i = 0, meta_data_id = 0;i<data.size();meta_data_id++) {
        ret.pb(vector<Data>());
        for (int k = 0; k < subDataset_size && i < data.size(); i++, k++) {
            ret[meta_data_id].pb(data[i]);
        }
    }
    return ret;
}

secondOrderDatasets<Data> initSecondOrderDatasets(firstOrderDatasets<Data> data)
{
    secondOrderDatasets<Data> ret;
    ret.n = data.n;
    int subDataset_size = 2;
    ret.train_meta_data = initSecondOrderDataset(data.train_data, subDataset_size);
    ret.test_meta_data = initSecondOrderDataset(data.test_data, subDataset_size);

    return ret;
}

template<typename datatype>
class bitvectorFunctionSolver: public secondOrderLearning<datatype>
{
    typedef secondOrderLearning<datatype> base;
public:
    int root_iter = 60, leaf_iter = 60;

    bitvectorFunctionSolver(int _leaf_iter): secondOrderLearning<datatype>()
    {
        assert(_leaf_iter >= 1);
        leaf_iter = _leaf_iter;
    }

    firstOrderDatasets<datatype> first_order_data;
    secondOrderDatasets<datatype> meta_data;

    bool first_order_data_inited = false;
    bool meta_data_inited = false;

    void initFirstOrderData(int n, firstOrderDatasets<datatype> _first_order_data)
    {
        first_order_data = _first_order_data;
        first_order_data_inited = true;
    }

    void initMetaData(int n, firstOrderDatasets<datatype> _first_order_data)
    {


        first_order_data = _first_order_data;
        first_order_data_inited = true;

        meta_data = initSecondOrderDatasets(first_order_data);

        meta_data_inited = true;
    }

    const double treshold = 0.4;
    double min_treshold = treshold/3;

    net_and_score train_to_order_by_difficulty(int n, bool print)
    {
        assert(first_order_data_inited);

        net_and_score learner = net_and_score(net(n, 2*n, 1));

        order_tasks_by_difficulty_via_mutli_task_learning(learner, first_order_data, print, leaf_iter, treshold);
    }

    net_and_score train_to_learn(int n, bool print)
    {
        assert(first_order_data_inited);

        srand(time(0));
        net_and_score learner = net_and_score(net(n, 2*n, 1));

        return train_to_learn(n, learner, print);
    }

    net_and_score train_to_learn(int n, net_and_score learner, bool print)
    {
        assert(first_order_data_inited);

        base::meta_learn(learner, first_order_data.train_data, print, leaf_iter, treshold);

        min_treshold = learner.max_error;
        leaf_iter = learner.max_leaf_iter;

        assert(leaf_iter != 0);

        return learner;
    }


    void test_to_learn(net_and_score learner)
    {
        assert(first_order_data_inited);

        meta_net_and_score rez =
                evaluate_learner(learner, first_order_data.test_data, true, leaf_iter, min_treshold);

        learner.printWeights();
        cout << "rez = " << rez.print() <<endl;
    }

    meta_net_and_score train_to_meta_learn(int n)
    {
        assert(meta_data_inited);

        srand(time(0));
        meta_net_and_score learner = net_and_score(net(n, 2*n, 1));

        return train_to_meta_learn(n, learner);
    }

    meta_net_and_score train_to_meta_learn(int n, meta_net_and_score learner)
    {
        assert(meta_data_inited);

        base::learn_to_meta_learn(learner, meta_data.train_meta_data, root_iter, leaf_iter, treshold);

        min_treshold = learner.max_error;
        leaf_iter = learner.max_leaf_iter;

        assert(leaf_iter != 0);

        return learner;
    }

    void test_to_meta_learn(meta_net_and_score learner)
    {
        assert(meta_data_inited);
        meta_net_and_score rez =
            base::evaluate_meta_learner(learner, meta_data.test_meta_data, true, root_iter, leaf_iter, min_treshold);

        learner.printWeights();
        cout << "rez = " << rez.print() <<endl;
    }


    void print_test_data_ordering(int n, vector<net*> ensamble_nets)
    {

        assert(first_order_data_inited);

        vector<pair<DecisionTreeScore, int> > train_error_ordering;
        vector<pair<DecisionTreeScore, int> > opt_ordering;

        vector<DecisionTreeScore> opt_scores = vector<DecisionTreeScore>
                (first_order_data.test_data.size());

        for(int i = 0; i < first_order_data.test_data.size();i++)
        {
            //ensamble neural training error
            double local_error = 0;
            for (int j = 0; j < ensamble_nets.size(); j++) {
                assert(j < ensamble_nets.size());
                net leaf_learner = ensamble_nets[j];
                net::parameters leaf_parameters = cutoff_param(leaf_iter, 0.01);
                //cout << "init potential branch train" << endl;
                leaf_learner.train(&first_order_data.test_data[i], leaf_parameters, &net::softPriorityTrain);
                //cout << "end potential branch train" << endl;
                local_error += (leaf_learner.get_error_of_data(&first_order_data.test_data[i]));
            }

            //opt error
            dp_decision_tree<datatype> decision_tree_solver;
            net::parameters cutoff_parametes;
            DecisionTreeScore size_opt = decision_tree_solver.synthesize_decision_tree_and_get_size
                    (cutoff_parametes, n, first_order_data.test_data[i], optimal);

            train_error_ordering.pb(mp(DecisionTreeScore(local_error), i));
            opt_ordering.pb(mp(size_opt, i));
            opt_scores[i] = size_opt;
        }
        sort_v(opt_ordering);
        sort_v(train_error_ordering);
        for(int i = 0;i<opt_ordering.size();i++)
        {
            cout << first_order_data.test_data[train_error_ordering[i].s].printConcatinateOutput() <<"\t";
            cout << fixed << setprecision(4) << (double)train_error_ordering[i].f.size << "\t | \t";

            cout << first_order_data.test_data[train_error_ordering[i].s].printConcatinateOutput() <<"\t";
            cout << (int)opt_scores[train_error_ordering[i].s].size <<"\t";
            cout << (int)opt_scores[train_error_ordering[i].s].num_solutions << "\t | \t";

            cout << endl;
        }
        cout << endl;
    }

    /*meta_net_and_score learn_to_generalize(meta_net_and_score learner)
    {

    }*/
};

class latentDecisionTreeExtractor: public secondOrderLearning<Data>
{
public:

    latentDecisionTreeExtractor(): secondOrderLearning()
    {
    }
    
    class SynhtesizerScoreCoparitor
    {
    public:
        int sum_distance = 0;
        double sum_ratio = 0;
        double max_ratio = 0;
        int max_distance = 0;
        int count = 0;

        void clear_score()
        {
            sum_distance = 0;
            sum_ratio = 0;

            max_ratio = 0;
            max_distance = 0;
            count = 0;
        }

        void compose_with(DecisionTreeScore _base, DecisionTreeScore _other)
        {
            int base = _base.size;
            int other = _other.size;

            int distance = other - base;
            double ratio = (double)other/base;
            sum_distance+=(other-base);
            assert(base != 0);
            sum_ratio += ratio;
            count++;

            max_distance = max(max_distance, distance);
            max_ratio = max(ratio, max_ratio);
        }

        string print()
        {
            return "sum_dist = \t" + std::to_string(sum_distance) +
            "\t, max_dist = \t" + std::to_string(max_distance) +
            "\t| avg_ratio = \t" + std::to_string(sum_ratio/count) +
            "\t, max_ratio = \t" + std::to_string(max_ratio);
        }
    };
    
    class SynthesizerScoreSum
    {
    public:
        int sum_sizes = 0;

        void clear_score()
        {
            sum_sizes = 0;
        }

        string print()
        {
            return "sum = \t" + std::to_string(sum_sizes);
        }

        string simple_print()
        {
            return  std::to_string(sum_sizes);
        }

        void compose_with(DecisionTreeScore new_size)
        {
            sum_sizes+=new_size.size;
        }

        bool operator < (const SynthesizerScoreSum& other) const
        {
            return sum_sizes < other.sum_sizes;
        }
    };
    
    class SynthesizersScoreSummary: public net
    {
    public:
        SynthesizerScoreSum base_synthesizer_score;
        SynthesizerScoreSum other_synthesizer_score;
        SynhtesizerScoreCoparitor comparitor;

        SynthesizersScoreSummary(){}

        SynthesizersScoreSummary(net self): net(self)
        {

        }
        
        void compose_with(DecisionTreeScore base_score, DecisionTreeScore other_score)
        {
            base_synthesizer_score.compose_with(base_score);
            other_synthesizer_score.compose_with(other_score);
            comparitor.compose_with(base_score, other_score);

        }

        int sum_errors()
        {
            return comparitor.sum_distance;
        }

        void clear_score()
        {
            base_synthesizer_score.clear_score();
            other_synthesizer_score.clear_score();
            comparitor.clear_score();
        }

        string print_other_synthesizer()
        {
            return "compare_with = {\t" + other_synthesizer_score.print() + "\t}";
        }

        string print()
        {
            return
                "base = {\t" + base_synthesizer_score.print() +
                "\t}; compare_with = {\t" + other_synthesizer_score.print()+
                "\t}; comparitor = {\t" + comparitor.print() + "\t}";
        }

        string print_base()
        {
            return base_synthesizer_score.simple_print();
        }

        string print_other()
        {
            return other_synthesizer_score.simple_print();
        }

        bool operator < (const SynthesizersScoreSummary& other) const
        {
            return base_synthesizer_score < other.base_synthesizer_score;
        }
    };

    /*
    class SynthesizersScoreSummary: public net
    {
    public:
        double max_ratio = 1;
        int max_delta = 0;
        int sum_sizes = 0;
        int sum_errors = 0;
        int num_non_opt = 0;
        int num_neural_guided_non_opt = 0;
        
        vector<Data> non_opt;

        int sum_entropy_errors = 0;
        int num_entropy_non_opt = 0;

        SynthesizersScoreSummary()
        {

        }

        SynthesizersScoreSummary(net self): net(self)
        {

        }

        bool operator < (const SynthesizersScoreSummary& ret) const
        {
            return sum_errors < ret.sum_errors;
        }

        string print_sums()
        {
            return ""
        }
        
        string print()
        {
            return "sum_errors = \t" + std::to_string(sum_errors)+
                   ", num_non_opt = " + std::to_string(num_non_opt);
        }

        string print_entropy()
        {
            return "sum_entropy_errors = \t" + std::to_string(sum_entropy_errors)+
                   ", num_entropy_non_opt = " + std::to_string(num_entropy_non_opt);
        }

        void clear_score()
        {
            max_ratio = 1;
            max_delta = 0;
            sum_sizes = 0;
            sum_errors = 0;
            num_non_opt = 0;

            non_opt.clear();

            sum_entropy_errors = 0;
            num_entropy_non_opt = 0;
        }
    };
     */

    SynthesizersScoreSummary run_decision_tree_synthesis_comparison(
            SynthesizersScoreSummary meta_meta_net,
            bitvectorFunctionSolver<Data> bitvector_data,
            int n,
            vector<vector<Data> > data,
            bool run_reptile,
            bool print)
    {
        SynthesizersScoreSummary ret = meta_meta_net;
        ret.clear_score();
        for(int i = 0;i<data.size();i++)
        {
            SynthesizersScoreSummary meta_net = meta_meta_net;
            if(run_reptile)
            {
                reptile_train(meta_net, data[i], bitvector_data.root_iter, bitvector_data.leaf_iter, bitvector_data.min_treshold);
            }

            for(int j = 0;j<data[i].size();j++)
            {
                /*SynthesizersScoreSummary leaf_net = meta_net;

                net::parameters param = cutoff_param(8, 0.01);
                int local_iter = leaf_net.train(&data[i][j], param, &net::softPriorityTrain);

                double local_error = leaf_net.get_error_of_data(&data[i][j]);

                if(print)
                {
                    cout << "f: " << data[i][j].printConcatinateOutput() << endl;
                    cout << "iters: " << local_iter <<endl;
                    cout << "error: " << local_error << endl;
                }*/


                SynthesizersScoreSummary new_leaf_net = meta_net;

                //tailor is the neural network that tailors towards inducing a special structure
                net::parameters tailor_param = cutoff_param(7, 0.01);
                tailor_param.track_dimension_model = true;
                tailor_param.neural_net = &new_leaf_net;

                dp_decision_tree<Data> decision_tree_solver;
                DecisionTreeScore size;
                assert(0);
                /*
                 * NEED TO REFACTOR WAY OF CALLING THE DECISION TREE SYNTHESISER
                 * NEED TO PUT VALUE ON size
                 * =
                        decision_tree_solver.old_extract_decision_tree_and_compare_with_opt_and_entropy(tailor_param, n, data[i][j]);

                if(print)
                {
                    cout << "f: " << data[i][j].printConcatinateOutput() << endl;
                    cout << "opt = \t" << size.get_opt() << "\t; my = " << size.neural_guided_size << "\t; entropy = "
                         << size.entropy_guided_size << endl << endl;
                }

                if(size.opt_defined) 
                {
                    assert(0);//update ret summary
                    / *if (size.neural_guided_size > size.get_opt()) {
                        ret.num_non_opt++;
                        ret.max_ratio = max(ret.max_ratio, (double) size.neural_guided_size / size.get_opt());
                        ret.max_delta = max(ret.max_delta, size.neural_guided_size - size.get_opt());
                        ret.sum_errors += size.neural_guided_size - size.get_opt();
                        ret.sum_sizes += size.neural_guided_size;

                        ret.non_opt.pb(data[i][j]);
                    }
                    if (size.entropy_guided_size > size.get_opt()) {
                        ret.num_entropy_non_opt++;
                        ret.sum_entropy_errors += size.entropy_guided_size - size.get_opt();
                    }* /
                }
                else
                {
                    //need to handle case when opt is not defiend
                    assert(0);
                }*/
            }
        }
        return ret;
    }

    void SA_over_latent_decision_trees(
            SynthesizersScoreSummary &best,
            bitvectorFunctionSolver<Data> bitvector_data,
            int n,
            vector<vector<Data> > data,
            bool run_reptile_training)
    {
        double temperature = 1;
        int iter_count = 15;

        SynthesizersScoreSummary at = best = run_decision_tree_synthesis_comparison
                (best, bitvector_data, n, data, run_reptile_training, false);

        for(int iter = 0; iter<iter_count; iter++, temperature-=(1.0/iter_count))
        {
            //at.printWeights();

            SynthesizersScoreSummary next = net_and_score(step(at, 0.35*temperature));

            //next.printWeights();

            SynthesizersScoreSummary next_score = run_decision_tree_synthesis_comparison
                    (next, bitvector_data, n, data, run_reptile_training, false);

            cout << "iter = " << iter << "; at:: " << at.print() <<"; next_score:: "<< next_score.print() <<endl;

            if(next_score < best)
            {
                at = next_score;
                best = next_score;
            }
            else if(next_score < at)
            {
                at = next_score;
            }
            else if(100*(double)(next_score.sum_errors()-at.sum_errors())/at.sum_errors() < (double)rand(0, (int)(25*temperature)))
            {
                cout << "take" <<endl;
                at = next_score;
            }
        }
    }

    void train(int n)
    {
        cout << "IN latentDecisionTreeExtractor" <<endl <<endl;

        int leaf_iter = 30;

        bitvectorFunctionSolver<Data> bitvector_data(leaf_iter);

        //meta_net_and_score meta_meta_net = bitvector_data.train(n);
        SynthesizersScoreSummary meta_meta_net = net_and_score(net(n, 2*n, 1));

        assert(n<=4);
        bitvector_data.initMetaData(n, completeFirstOrderTestDataset<Data>(n));

        vector<vector<Data> > train_meta_data = bitvector_data.meta_data.train_meta_data;
        vector<vector<Data> > test_meta_data = bitvector_data.meta_data.test_meta_data;

        //vector<vector<Data> > harder_data = train_meta_data;

        /*for(int i = 0;i<1;i++)
        {
            vector<Data> step_harder_data = run_decision_tree_synthesis_comparison
                    (meta_meta_net, bitvector_data, n, harder_data, bitvector_data.treshold, false, false).non_opt;
            cout << step_harder_data.size() <<endl;
            harder_data = initSecondOrderDataset(step_harder_data, 2);
        }*/

        bool pre_train_meta_init = true;

        if(pre_train_meta_init) {

            bitvector_data.meta_data.train_meta_data.clear();
            int rand_step = train_meta_data.size()/3;
            for (int i = rand((int)(rand_step*0.5), (int)(1.5*rand_step));
                 i < train_meta_data.size();
                 i+=rand((int)(rand_step*0.5), (int)(1.5*rand_step)))
            {
                bitvector_data.meta_data.train_meta_data.pb(train_meta_data[i]);
            }
            bitvector_data.meta_data.test_meta_data = train_meta_data;


            meta_meta_net = bitvector_data.train_to_meta_learn(n);
            bitvector_data.test_to_meta_learn(meta_net_and_score(meta_meta_net));

        }
        /*int prev_leaf_iter = bitvector_data.leaf_iter;
        for(bitvector_data.leaf_iter = 1; bitvector_data.leaf_iter < 16;bitvector_data.leaf_iter++) {
            SynthesizersScoreSummary local_meta_meta_net = meta_meta_net;
            SynthesizersScoreSummary at = run_decision_tree_synthesis_comparison
                    (local_meta_meta_net, bitvector_data, n, test_meta_data, pre_train_meta_init, false);

            cout << "leaf_iter = " << bitvector_data.leaf_iter << ", my_error = " << at.print() << endl;
            cout << "entropy_error = " << at.print_entropy() <<endl <<endl;
        }
        bitvector_data.leaf_iter = prev_leaf_iter;
        */
        SynthesizersScoreSummary at = run_decision_tree_synthesis_comparison
                (meta_meta_net, bitvector_data, n, test_meta_data, pre_train_meta_init, true);

        cout << "BEFORE TRAIN: at = " << at.print() << endl;
        cout << "Entropy = " << at.print_other_synthesizer() <<endl <<endl;

        SA_over_latent_decision_trees(meta_meta_net, bitvector_data, n, train_meta_data, pre_train_meta_init);

        cout << "In main train(..)" <<endl;
        cout <<"meta_meta_net:: " << meta_meta_net.print() <<endl;


        if(pre_train_meta_init)
        {
            cout << "meta_meta_net after train:" << endl;
            bitvector_data.test_to_meta_learn(meta_net_and_score(meta_meta_net));
        }
        SynthesizersScoreSummary final_rez = run_decision_tree_synthesis_comparison
                (meta_meta_net, bitvector_data, n, test_meta_data, pre_train_meta_init, true);

        cout << "AFTER TRAIN: at = " << final_rez.print() << endl;
        cout << "entropy = " << final_rez.print_other_synthesizer() <<endl <<endl;

        cout << "END" <<endl;


    }


    /*SynthesizersScoreSummary run_new_decision_tree_synthesis_comparison(
            vector<net*> progressive_nets,
            bitvectorFunctionSolver bitvector_data,
            int n,
            vector<vector<Data> > data,
            bool print)
    {
        SynthesizersScoreSummary ret;
        ret.clear_score();

        for(int i = 0;i<data.size();i++)
        {
            for(int j = 0;j<data[i].size();j++)
            {
                net::parameters tailor_param = meta_cutoff(bitvector_data.root_iter, bitvector_data.leaf_iter);
                tailor_param.progressive_nets = progressive_nets;

                dp_decision_tree decision_tree_solver;
                DecisionTreeScore size =
                        decision_tree_solver.new_extract_decision_tree_and_compare_with_opt_and_entropy(tailor_param, n, data[i][j]);

                if(print)
                {
                    cout << "f: " << data[i][j].printConcatinateOutput() << endl;
                    cout << "opt = \t" << size.opt << "\t; my = " << size.neural_guided_size << "\t; entropy = "
                         << size.entropy_guided_size << endl << endl;
                }

                if(size.neural_guided_size > size.opt)
                {
                    ret.num_non_opt++;
                    ret.max_ratio = max(ret.max_ratio, (double)size.neural_guided_size/size.opt);
                    ret.max_delta = max(ret.max_delta, size.neural_guided_size - size.opt);
                    ret.sum_errors += size.neural_guided_size - size.opt;
                    ret.sum_sizes += size.neural_guided_size;

                    ret.non_opt.pb(data[i][j]);
                }
                if(size.entropy_guided_size > size.opt)
                {
                    ret.num_entropy_non_opt++;
                    ret.sum_entropy_errors += size.entropy_guided_size - size.opt;
                }
            }
        }
        return ret;
    }*/

    vector<int> hidden_layer_block(int width, int depth)
    {
        vector<int> ret;
        for(int i = 0;i<depth;i++)
        {
            ret.pb(width);
        }
        return ret;
    }

    template<typename datatype>
    void train_library(
            int min_trainable_n,
            int max_n, vector<int> leaf_iters, vector<int> hidden_layer_width, vector<int> hidden_layer_depth,
            int num_ensambles, vector<firstOrderDatasets<datatype> > first_order_datasets)
    {
        assert(leaf_iters.size() > max_n-1);
        assert(hidden_layer_width.size() > max_n-1);
        net ensamble_progressive_nets[10][10];
        vector<vector<net*> > ensamble_progressive_net_pointers;
        ensamble_progressive_net_pointers.pb(vector<net*>());// for n = 0;

        for(int local_n = 2; local_n< min_trainable_n; local_n++)
        {
            vector<net*> local_ensamble;
            for(int ensamble_id = 0; ensamble_id < num_ensambles; ensamble_id++) {
                srand(time(0));

                vector<int> hidden_layer_widths = hidden_layer_block(hidden_layer_width[local_n - 1], hidden_layer_depth[local_n-1]);

                SynthesizersScoreSummary meta_meta_net =
                        net_and_score(net(local_n - 1, hidden_layer_widths, 1));

                ensamble_progressive_nets[local_n - 1][ensamble_id] = meta_meta_net;
                local_ensamble.pb(&ensamble_progressive_nets[local_n - 1][ensamble_id]);
            }
            ensamble_progressive_net_pointers.pb(local_ensamble);
        }

        for(int local_n = min_trainable_n; local_n <= max_n; local_n++)
        {

            vector<net*> local_ensamble;


            for(int ensamble_id = 0; ensamble_id < num_ensambles; ensamble_id++)
            {
                srand(time(0));

                vector<int> hidden_layer_widths = hidden_layer_block(hidden_layer_width[local_n - 1], hidden_layer_depth[local_n-1]);

                SynthesizersScoreSummary meta_meta_net =
                        net_and_score(net(local_n - 1, hidden_layer_widths, 1));

                bool do_learn = true;

                bitvectorFunctionSolver<datatype> sub_bitvector_data(leaf_iters[local_n - 1]);

                if(do_learn)
                {

                    sub_bitvector_data.initFirstOrderData(local_n - 1, first_order_datasets[local_n-1]);

                    local_ensamble.pb(&meta_meta_net);
                    sub_bitvector_data.print_test_data_ordering(local_n-1, local_ensamble);
                    local_ensamble.clear();

                    meta_meta_net = sub_bitvector_data.train_to_learn(local_n - 1, false); //false refers to print

                }
                else
                {
                    sub_bitvector_data.initFirstOrderData(local_n - 1, completeFirstOrderTestDataset<datatype>(local_n - 1));
                }

                local_ensamble.pb(&meta_meta_net);
                sub_bitvector_data.print_test_data_ordering(local_n-1, local_ensamble);
                local_ensamble.clear();

                //sub_bitvector_data.test_to_learn(meta_net_and_score(meta_meta_net));

                ensamble_progressive_nets[local_n - 1][ensamble_id] = meta_meta_net;
                local_ensamble.pb(&ensamble_progressive_nets[local_n - 1][ensamble_id]);
            }

            ensamble_progressive_net_pointers.pb(local_ensamble);

            net::parameters cutoff_parameter = net::parameters(leaf_iters);
            cutoff_parameter.ensamble_progressive_nets = ensamble_progressive_net_pointers;

            bitvectorFunctionSolver<datatype> complete_bitvector_data(leaf_iters[local_n]);

            complete_bitvector_data.initFirstOrderData(local_n, first_order_datasets[local_n]);


            dp_decision_tree<datatype> decision_tree_solver;

            SynthesizersScoreSummary opt_to_entropy;
            SynthesizersScoreSummary opt_to_neural;
            SynthesizersScoreSummary opt_to_random;
            SynthesizersScoreSummary neural_to_entropy;
            SynthesizersScoreSummary random_to_entropy;


            for(int i = 0;i < complete_bitvector_data.first_order_data.test_data.size();i++) {

                DecisionTreeScore size_opt, size_neural_guided, size_entropy_guided, size_random_guided;


                if(local_n <= 4)
                {

                    size_opt = decision_tree_solver.synthesize_decision_tree_and_get_size
                            (cutoff_parameter, local_n, complete_bitvector_data.first_order_data.test_data[i], optimal);
                }


                size_neural_guided = decision_tree_solver.synthesize_decision_tree_and_get_size
                        (cutoff_parameter, local_n, complete_bitvector_data.first_order_data.test_data[i], neural_guided);


                size_entropy_guided = decision_tree_solver.synthesize_decision_tree_and_get_size
                        (cutoff_parameter, local_n, complete_bitvector_data.first_order_data.test_data[i], entropy_guided);

                //size_random_guided = decision_tree_solver.synthesize_decision_tree_and_get_size
                //        (cutoff_parameter, local_n, complete_bitvector_data.first_order_data.test_data[i], random_guided);

                //cout parameters:


                //cout << complete_bitvector_data.first_order_data.test_data[i].printConcatinateOutput() << endl;
                //cout << decision_tree_size.print() << endl << endl;



                if(local_n <= 4)
                {
                    opt_to_entropy.compose_with(size_opt, size_entropy_guided);
                    opt_to_neural.compose_with(size_opt, size_neural_guided);
                    //opt_to_random.compose_with(size_opt, size_random_guided);
                }
                
                neural_to_entropy.compose_with(size_neural_guided, size_entropy_guided);
                //random_to_entropy.compose_with(size_random_guided, size_entropy_guided);

            }
            cout << endl;

            fout << num_ensambles << "\t" << opt_to_neural.print_base() << "\t" << neural_to_entropy.print_base() <<"\t" << neural_to_entropy.print_other() <<endl;

            cout << "ensamble_size = " << num_ensambles << endl;
            cout << "hidden_layer_width:" <<endl;
            for(int j = 2; j < local_n; j++)
            {
                cout << "n_in = " << j << "\thidden_width "<< hidden_layer_width[j] <<endl;
            }
            cout << "leaf_iters:" <<endl;
            for(int j = 2; j < local_n; j++)
            {
                cout << "n_in = " << j << "\tleaf_iter "<< leaf_iters[j] <<endl;
            }
            if(local_n <= 4 && true)
            {
                cout << "opt_to_neural:: " << opt_to_neural.print() << endl;
                //cout << "opt_to_entropy:: " << opt_to_entropy.print() << endl;
                //cout << "opt_to_random:: " << opt_to_random.print() << endl;
            }
            cout << "neural_to_entropy:: " << neural_to_entropy.print() << endl;
            //cout << "random_to_entropy:: " << random_to_entropy.print() << endl <<endl;
        }
    }
};

/*
11111111	0.0677	 | 	11111111	1	 |
00110011	0.0894	 | 	00110011	3	 |
01010101	0.0899	 | 	01010101	3	 |
00001111	0.0934	 | 	00001111	3	 |
11001100	0.0941	 | 	11001100	3	 |
11110000	0.0972	 | 	11110000	3	 |
10101010	0.0986	 | 	10101010	3	 |
11110011	0.1305	 | 	11110011	5	 |
01110111	0.1311	 | 	01110111	5	 |
10111011	0.1332	 | 	10111011	5	 |
11011101	0.1348	 | 	11011101	5	 |
00000101	0.1447	 | 	00000101	5	 |
01011111	0.1457	 | 	01011111	5	 |
00111111	0.1459	 | 	00111111	5	 |
11101110	0.1474	 | 	11101110	5	 |
10001000	0.1491	 | 	10001000	5	 |
10100000	0.1498	 | 	10100000	5	 |
00000011	0.1498	 | 	00000011	5	 |
11000000	0.1506	 | 	11000000	5	 |
11111100	0.1506	 | 	11111100	5	 |
11101000	0.1506	 | 	11101000	11	 |
00001100	0.1515	 | 	00001100	5	 |
00001010	0.1532	 | 	00001010	5	 |
00100010	0.1543	 | 	00100010	5	 |
11111010	0.1545	 | 	11111010	5	 |
01000100	0.1562	 | 	01000100	5	 |
00110000	0.1578	 | 	00110000	5	 |
01110001	0.1608	 | 	01110001	11	 |
00101011	0.1617	 | 	00101011	11	 |
01010000	0.1658	 | 	01010000	5	 |
11010100	0.1669	 | 	11010100	11	 |
10110010	0.1676	 | 	10110010	11	 |
01001101	0.1699	 | 	01001101	11	 |
10001110	0.1728	 | 	10001110	11	 |
00010111	0.1746	 | 	00010111	11	 |
11001000	0.2633	 | 	11001000	7	 |
00001000	0.2658	 | 	00001000	7	 |
01000000	0.2742	 | 	01000000	7	 |
10101011	0.2758	 | 	10101011	7	 |
11111110	0.2773	 | 	11111110	7	 |
11111110	0.2773	 | 	11111110	7	 |
00100000	0.2798	 | 	00100000	7	 |
00001110	0.2847	 | 	00001110	7	 |
11011100	0.2879	 | 	11011100	7	 |
00110010	0.2956	 | 	00110010	7	 |
01010111	0.2997	 | 	01010111	7	 |
10111010	0.3002	 | 	10111010	7	 |
01010100	0.3035	 | 	01010100	7	 |
11110111	0.3056	 | 	11110111	7	 |
01111111	0.3078	 | 	01111111	7	 |
00000100	0.3110	 | 	00000100	7	 |
11111000	0.3136	 | 	11111000	7	 |
11111011	0.3163	 | 	11111011	7	 |
11101100	0.3183	 | 	11101100	7	 |
00000111	0.3213	 | 	00000111	7	 |
10001111	0.3224	 | 	10001111	7	 |
01001100	0.3260	 | 	01001100	7	 |
01001111	0.3302	 | 	01001111	7	 |
00010101	0.3313	 | 	00010101	7	 |
01110101	0.3319	 | 	01110101	7	 |
11010101	0.3320	 | 	11010101	7	 |
11011111	0.3323	 | 	11011111	7	 |
11001110	0.3386	 | 	11001110	7	 |
00010000	0.3470	 | 	00010000	7	 |
01110000	0.3495	 | 	01110000	7	 |
00001011	0.3495	 | 	00001011	7	 |
10001010	0.3501	 | 	10001010	7	 |
00111011	0.3504	 | 	00111011	7	 |
00000010	0.3509	 | 	00000010	7	 |
00101111	0.3582	 | 	00101111	7	 |
10110011	0.3632	 | 	10110011	7	 |
10101000	0.3701	 | 	10101000	7	 |
11110010	0.3709	 | 	11110010	7	 |
00010011	0.3718	 | 	00010011	7	 |
00110111	0.3727	 | 	00110111	7	 |
11001101	0.3745	 | 	11001101	7	 |
01010001	0.3748	 | 	01010001	7	 |
10001100	0.3750	 | 	10001100	7	 |
10100010	0.3768	 | 	10100010	7	 |
10111111	0.3831	 | 	10111111	7	 |
01000101	0.3862	 | 	01000101	7	 |
11010000	0.3976	 | 	11010000	7	 |
00100011	0.4062	 | 	00100011	7	 |
00110001	0.4093	 | 	00110001	7	 |
11110100	0.4110	 | 	11110100	7	 |
11000100	0.4138	 | 	11000100	7	 |
10101110	0.4186	 | 	10101110	7	 |
00001101	0.4228	 | 	00001101	7	 |
11100000	0.4428	 | 	11100000	7	 |
11111101	0.4634	 | 	11111101	7	 |
01110011	0.4897	 | 	01110011	7	 |
11101111	0.4920	 | 	11101111	7	 |
01011101	0.5359	 | 	01011101	7	 |
10000001	0.5969	 | 	10000001	11	 |
00001001	0.6024	 | 	00001001	9	 |
10100110	0.6128	 | 	10100110	11	 |
01101010	0.6334	 | 	01101010	11	 |
10011000	0.6411	 | 	10011000	9	 |
10111110	0.6464	 | 	10111110	9	 |
00101101	0.6468	 | 	00101101	11	 |
10111001	0.6481	 | 	10111001	9	 |
01110010	0.6728	 | 	01110010	7	 |
00111000	0.6802	 | 	00111000	9	 |
10100011	0.6812	 | 	10100011	7	 |
11101011	0.6817	 | 	11101011	9	 |
11011110	0.6915	 | 	11011110	9	 |
10100100	0.6932	 | 	10100100	9	 |
10001101	0.6961	 | 	10001101	7	 |
10101100	0.7017	 | 	10101100	7	 |
00100001	0.7095	 | 	00100001	9	 |
01111010	0.7105	 | 	01111010	9	 |
11100100	0.7110	 | 	11100100	7	 |
11100011	0.7117	 | 	11100011	9	 |
11011011	0.7141	 | 	11011011	11	 |
01101100	0.7143	 | 	01101100	11	 |
01100110	0.7143	 | 	01100110	7	 |
01100100	0.7143	 | 	01100100	9	 |
11110110	0.7163	 | 	11110110	9	 |
01111000	0.7165	 | 	01111000	11	 |
11100110	0.7170	 | 	11100110	9	 |
01100001	0.7215	 | 	01100001	13	 |
00011000	0.7271	 | 	00011000	11	 |
01001010	0.7274	 | 	01001010	9	 |
01000010	0.7274	 | 	01000010	11	 |
00111100	0.7329	 | 	00111100	7	 |
00011110	0.7356	 | 	00011110	11	 |
00101110	0.7371	 | 	00101110	7	 |
10101101	0.7375	 | 	10101101	9	 |
00111010	0.7398	 | 	00111010	7	 |
01101011	0.7402	 | 	01101011	13	 |
00011010	0.7446	 | 	00011010	9	 |
10101001	0.7447	 | 	10101001	11	 |
10010100	0.7470	 | 	10010100	13	 |
01011000	0.7478	 | 	01011000	9	 |
11000001	0.7502	 | 	11000001	9	 |
00101100	0.7511	 | 	00101100	9	 |
10010111	0.7516	 | 	10010111	13	 |
10010011	0.7516	 | 	10010011	11	 |
11100001	0.7535	 | 	11100001	11	 |
11100101	0.7535	 | 	11100101	9	 |
11100111	0.7593	 | 	11100111	11	 |
10010010	0.7606	 | 	10010010	13	 |
10011010	0.7606	 | 	10011010	11	 |
10110001	0.7615	 | 	10110001	7	 |
01000001	0.7638	 | 	01000001	9	 |
11010111	0.7715	 | 	11010111	9	 |
01011100	0.7715	 | 	01011100	7	 |
00011001	0.7727	 | 	00011001	9	 |
01100011	0.7730	 | 	01100011	11	 |
10110101	0.7738	 | 	10110101	9	 |
00110100	0.7746	 | 	00110100	9	 |
00100100	0.7746	 | 	00100100	11	 |
11101001	0.7777	 | 	11101001	13	 |
10111000	0.7791	 | 	10111000	7	 |
00101001	0.7792	 | 	00101001	13	 |
00011011	0.7794	 | 	00011011	7	 |
01011011	0.7801	 | 	01011011	9	 |
00111001	0.7805	 | 	00111001	11	 |
00100110	0.7807	 | 	00100110	9	 |
01101111	0.7815	 | 	01101111	9	 |
01110100	0.7847	 | 	01110100	7	 |
10010101	0.7877	 | 	10010101	11	 |
00100101	0.7893	 | 	00100101	9	 |
10000100	0.7911	 | 	10000100	9	 |
10011101	0.7912	 | 	10011101	9	 |
10010000	0.7926	 | 	10010000	9	 |
01101000	0.7956	 | 	01101000	13	 |
10100111	0.7970	 | 	10100111	9	 |
01001011	0.8007	 | 	01001011	11	 |
11011010	0.8075	 | 	11011010	9	 |
10000010	0.8092	 | 	10000010	9	 |
10011110	0.8099	 | 	10011110	13	 |
00100111	0.8109	 | 	00100111	7	 |
01101001	0.8134	 | 	01101001	15	 |
11011001	0.8170	 | 	11011001	9	 |
10001001	0.8177	 | 	10001001	9	 |
01001110	0.8182	 | 	01001110	7	 |
10010001	0.8215	 | 	10010001	9	 |
10011001	0.8224	 | 	10011001	7	 |
11111001	0.8240	 | 	11111001	9	 |
10111101	0.8271	 | 	10111101	11	 |
00011101	0.8329	 | 	00011101	7	 |
00111110	0.8425	 | 	00111110	9	 |
11100010	0.8428	 | 	11100010	7	 |
11011000	0.8449	 | 	11011000	7	 |
10000111	0.8465	 | 	10000111	11	 |
10000011	0.8465	 | 	10000011	9	 |
01100010	0.8467	 | 	01100010	9	 |
11001001	0.8467	 | 	11001001	11	 |
11001011	0.8509	 | 	11001011	9	 |
10111100	0.8541	 | 	10111100	9	 |
00110101	0.8543	 | 	00110101	7	 |
01010011	0.8564	 | 	01010011	7	 |
00101000	0.8566	 | 	00101000	9	 |
01111100	0.8595	 | 	01111100	9	 |
10110111	0.8622	 | 	10110111	9	 |
01111110	0.8658	 | 	01111110	11	 |
01010010	0.8674	 | 	01010010	9	 |
01001001	0.8679	 | 	01001001	13	 |
11010010	0.8687	 | 	11010010	11	 |
01100111	0.8737	 | 	01100111	9	 |
01000110	0.8775	 | 	01000110	9	 |
11000110	0.8775	 | 	11000110	11	 |
10000110	0.8775	 | 	10000110	13	 |
11000011	0.8793	 | 	11000011	7	 |
01000011	0.8816	 | 	01000011	9	 |
01111001	0.8837	 | 	01111001	13	 |
00010110	0.8846	 | 	00010110	13	 |
10010110	0.8846	 | 	10010110	15	 |
01010110	0.8846	 | 	01010110	11	 |
11010110	0.8846	 | 	11010110	13	 |
10100101	0.8922	 | 	10100101	7	 |
10100001	0.8922	 | 	10100001	9	 |
10011111	0.8947	 | 	10011111	9	 |
01111101	0.8955	 | 	01111101	9	 |
11010001	0.8966	 | 	11010001	7	 |
01101101	0.8969	 | 	01101101	13	 |
01100101	0.8976	 | 	01100101	11	 |
01100000	0.8982	 | 	01100000	9	 |
11001010	0.8994	 | 	11001010	7	 |
01111011	0.9038	 | 	01111011	9	 |
00111101	0.9039	 | 	00111101	9	 |
00110110	0.9107	 | 	00110110	11	 |
10011011	0.9126	 | 	10011011	9	 |

Network =
{
    {
        { {0.343846, 0.878073, -0.599613}, {-0.287050} },
        { {-0.038455, -0.055753, -0.045319}, {0.308151} },
        { {0.121851, -0.408460, -0.550222}, {-0.011402} },
        { {0.018635, 0.018354, 0.006582}, {0.162808} },
        { {-0.097966, -0.053717, 0.064828}, {0.048662} },
        { {-0.459572, -0.317840, -0.487781}, {0.378314} }
    },
    {
        { {0.584033, -6.756721, 0.167906, 3.677181, 3.568711, -0.948333}, {0.419629} }
    }
}
rez = max_error = 	0.912614	 max_iter = 15
ensamble_size = 1
hidden_layer_width:
n_in = 2	hidden_width 4
n_in = 3	hidden_width 6
leaf_iters:
n_in = 2	leaf_iter 8
n_in = 3	leaf_iter 16
opt_to_neural:: base = {	sum = 	918	}; compare_with = {	sum = 	992	}; comparitor = {	sum_dist = 	74	, max_dist = 	6	| avg_ratio = 	1.081688	, max_ratio = 	1.461538	}
neural_to_entropy:: base = {	sum = 	992	}; compare_with = {	sum = 	1074	}; comparitor = {	sum_dist = 	82	, max_dist = 	12	| avg_ratio = 	1.110892	, max_ratio = 	2.333333	}

 */


class neural_decision_tree
{
public:

    Data the_Data;
    net the_teacher;

    vector<pair<bit_signature, int> > gate;

    neural_decision_tree* left = NULL;
    neural_decision_tree* right = NULL;

    neural_decision_tree* new_left = NULL;
    neural_decision_tree* new_right = NULL;

    int original_size = 0;
    int new_size = 0;

    neural_decision_tree()
    {

    }


    int init_root(int _n, net::parameters param, int (net::*training_f)(Data*, net::parameters param))
    {
        //string type = "input_is_output";
        //string type = "longest_substring_double_to_middle";
        //string type = "longest_substring_of_two_strings";
        string type = "longestSubstring_ai_is_1";
        //string type = "vector_loop_language";
        //string type = "a_i_is_a_i_plus_one_for_all";
        //string type = "longestSubstring_dai_di_is_0";

        int inputSize = _n, outputSize;

        the_Data.generateData(inputSize, outputSize, type);
        int ret1 = build_tree(param, training_f);
        return ret1;
    }

    int simple_build_tree(net::parameters param, int (net::*training_f)(Data*, net::parameters))
    {
        int inputSize = the_Data.numInputs;
        int outputSize = the_Data.numOutputs;
        int max_inter = inputSize;
        int hiddenLayers[10] =
                {
                        max_inter,
                        -1
                };

        the_teacher = net(inputSize, hiddenLayers, outputSize);

        //param.set_iteration_count(the_Data.size());

        bool expanded = false;

        original_size = 0;

        int ret = 0;
        int size_of_children = 0;

        assert(0);//check accuracy rate
        while(!the_teacher.test(&the_Data, 0.5).is_solved() && !expanded)
        {
            the_teacher.train(&the_Data, param, training_f);
            int the_bit = the_teacher.the_model.get_worst_dimension();

            if(the_bit != -1 || !the_Data.is_constant() /*|| !the_teacher.test(&the_Data, "no action")*/)
            {
                assert(ret == 0);
                expanded = true;

                if(the_bit == -1)
                {
                    the_bit = the_Data.get_first_active_input_bit();
                }

                gate.push_back(make_pair(the_bit, 1));

                left = new neural_decision_tree();
                right = new neural_decision_tree();

                the_Data.split(gate, left->the_Data, right->the_Data);



                assert(left->the_Data.size() > 0);
                assert(right->the_Data.size() > 0);

                bool enter = true;
                if(left->the_Data.size() >= 2)
                {
                    if(print_the_imporant_bit)cout << "l"<< the_bit <<endl;
                    size_of_children+=left->simple_build_tree(param, training_f);
                    original_size+=left->original_size;
                }
                else
                {
                    enter = false;
                    size_of_children+=1;
                    original_size++;
                }
                if(right->the_Data.size() >= 2)
                {
                    if(print_the_imporant_bit)cout << "r"<< the_bit <<endl;
                    size_of_children+=right->simple_build_tree(param, training_f);
                    original_size+=right->original_size;
                }
                else
                {
                    enter = false;
                    size_of_children+=1;
                    original_size++;
                }

            }
            else
            {
                if(print_the_imporant_bit)cout<<endl;
            }
            //cout << ret <<" "<< the_bit <<endl;
        }

        original_size+=1;
        ret+=1+size_of_children;
        return ret;

    }


    int build_tree(net::parameters param, int (net::*training_f)(Data*, net::parameters))
    {

        int improved = 0;
        int fail_to_improve = 0;
        int the_prev_bit = -1;

        int best_size = (1<<30);
        bit_signature best_bit;
        Data best_data = the_Data;
        bool is_constant = the_Data.is_constant();
        if(!is_constant)
        {
            for(int tries = 0, width = 3; tries<1;tries++)
            {
                Data local_data = the_Data;

                bit_signature the_bit;
                the_bit.vector_id = -1;

                if(false && local_data.num_active_input_bits()>1)
                {

                    Data local_local_data = local_data;
                    net first_teacher = net(local_local_data.numInputs, local_local_data.numOutputs);
                    first_teacher.train(&local_local_data, param, training_f);

                    typedef net::data_model::bit_dimension_pair bit_dimension_pair;
                    vector<bit_dimension_pair> sorted_pairs = first_teacher.the_model.sort_bit_dimension_pairs(&local_data); //first_teacher.the_model.sort_functional_dimension_pairs();//

                    int at = 0;
                    bool enter = false;
                    while(!enter && at < sorted_pairs.size())
                    {
                        bit_dimension_pair the_pair = sorted_pairs[at];
                        //cout << "Try :: :: " << the_pair.print() <<endl;
                        if(the_pair.val == 0)
                        {
                            break;
                        }
                        operator_signature new_operator = operator_signature(the_pair.f_bit, the_pair.s_bit, the_pair.f_dim, the_pair.s_dim);

                        if(print_try_and_finally)
                        {
                            cout << "Try :: :: " << the_pair.print() << endl;// << " " << bitset<4>(the_pair.the_gate.gate).to_string() << endl;
                        }
                        Data data_with_best_pair;
                        if(local_local_data.apply_new_operator_to_data(new_operator, data_with_best_pair))
                        {

                            operator_signature second_new_operator = operator_signature(!the_pair.f_bit, !the_pair.s_bit, the_pair.f_dim, the_pair.s_dim);

                            Data second_data_with_best_pair;
                            bool two_gates = false;
                            if(data_with_best_pair.apply_new_operator_to_data(second_new_operator, second_data_with_best_pair))
                            {
                                two_gates = true;
                                data_with_best_pair = second_data_with_best_pair;
                            }

                            net new_teacher = net(data_with_best_pair.numInputs, data_with_best_pair.numOutputs);
                            new_teacher.train(&data_with_best_pair, param, training_f);

                            int now_the_bit = new_teacher.the_model.get_worst_dimension();

                            if(now_the_bit == data_with_best_pair.numInputs-1 || now_the_bit == data_with_best_pair.numInputs-1-two_gates)
                            {
                                local_data = data_with_best_pair;

                                the_bit = local_data.circuit[now_the_bit];

                                Data data_with_single_kernel;
                                the_bit = local_data.add_single_kernel_to_base_and_discard_rest(the_bit, data_with_single_kernel);

                                local_data = data_with_single_kernel;



                                enter = true;
                                if(print_try_and_finally)
                                {
                                    cout << "Finally :: bit id =  " <<the_bit.vector_id<<" " << now_the_bit <<" : "<< the_pair.print() << " two gates: " << two_gates << endl;
                                }
                            }
                            else
                            {
                                //cout << "Fail :: :: " << the_pair.print() <<endl;
                                at++;
                            }
                        }
                        else
                        {
                            at++;
                        }
                    }

                }

                //if(the_bit.vector_id == -1)
                {
                    if(print_try_and_finally)
                    {
                        cout << "classic" <<endl;
                    }
                    net first_teacher = net(local_data.numInputs, local_data.numOutputs);
                    first_teacher.train(&local_data, param, training_f);
                    vector<bit_signature> bits_wanted = first_teacher.the_model.dimension_error_ratio_score;

                    for(int i = 0;i<bits_wanted.size();i++)
                    {
                        bits_wanted[i].vector_id = i;
                        double val = abs(bits_wanted[i].value);
                        if(isnan(val)) val = -1;
                        bits_wanted[i].set_sort_by(val);
                    }

                    sort_v(bits_wanted);
                    rev_v(bits_wanted);

                    Data data_with_all_kernels;
                    local_data.and_or_exstention(data_with_all_kernels, bits_wanted, width);
                    local_data = data_with_all_kernels;

                    the_bit = ensamble_teacher(&local_data, param.ensamble_size + 0*(local_data.numInputs/2+1), param, training_f).f;


                    Data data_with_single_kernel;
                    the_bit = local_data.add_single_kernel_to_base_and_discard_rest(the_bit, data_with_single_kernel);

                    local_data = data_with_single_kernel;

                }

                bit_signature the_local_final_bit = the_bit;

                int grown_children = 0;
                if(false)
                {
                    /*Data left_kernel, right_kernel;
                    gate.clear();
                    gate.push_back(mp(the_bit, 1));
                    local_data.split(gate, left_kernel, right_kernel);


                    assert(left_kernel.size() > 0);

                    assert(right_kernel.size() > 0);

                    neural_decision_tree* try_left = new neural_decision_tree();
                    neural_decision_tree* try_right = new neural_decision_tree();

                    //new_left = new neural_decision_tree();
                    //new_right = new neural_decision_tree();

                    try_left->the_Data = left_kernel;
                    try_right->the_Data = right_kernel;

                    //local_data.printData("here");

                    grown_children+=try_left->simple_build_tree(param, training_f);
                    grown_children+=try_right->simple_build_tree(param, training_f);
                     */
                }
                if(best_size >= grown_children)
                {
                    best_size = grown_children;

                    best_data = local_data;

                    best_bit = the_local_final_bit;
                }
                if(print_tree_synthesys)
                {
                    cout << local_data.size() << " ";
                    cout << " gate id: " << best_bit.vector_id;
                    if(best_bit.num_operators == 2)
                    {
                        cout << " " << bitset<4>(best_bit.gate).to_string() << "(" << best_bit.operands[0] <<", "<< best_bit.operands[1] <<")";

                    }
                    cout<< endl;
                }
            }

            Data left_kernel, right_kernel;
            gate.clear();
            gate.push_back(mp(best_bit, 1));
            best_data.split(gate, left_kernel, right_kernel);

            new_left = new neural_decision_tree();
            new_right = new neural_decision_tree();

            assert(left_kernel.size() > 0);

            assert(right_kernel.size() > 0);
            new_left->the_Data = left_kernel;
            new_right->the_Data = right_kernel;

            int grown_children = 0;
            grown_children+=new_left->build_tree(param, training_f);
            grown_children+=new_right->build_tree(param, training_f);

            the_Data = best_data;
        }
        else
        {
            assert(the_Data.size() > 0);
        }

        int grown_children = 0;
        //original_size = simple_build_tree(param, training_f);
        if(new_left != NULL && new_right != NULL)
        {
            grown_children = new_left->new_size + new_right->new_size;
        }
        else
        {
            grown_children = 0;
        }
        int ret = new_size = 1+grown_children;
        return ret;

    }

    void switch_off_synapse()
    {
        assert(0);
        int rand_layer = rand(0, the_teacher.size()-1);
        int rand_neuron = rand(0, the_teacher.layers[rand_layer].neurons.size()-1);
        int rand_disregard = rand(0, the_teacher.layers[rand_layer].neurons[rand_neuron].disregard.size()-1);
        int init_rand_neuron = rand_neuron;
        int init_rand_layer = rand_layer;
        while(false)//while not found what to switch off)
        {
            rand_neuron++;
            rand_neuron%=the_teacher.layers[rand_layer].size();
            if(rand_neuron == init_rand_neuron)
            {
                rand_layer++;
                rand_layer%=the_teacher.size();
                if(init_rand_layer == rand_layer)
                {
                    assert(0);
                }
            }
        }
        //damaged[rand_layer][rand_neuron] = true;
        //switch it off
        the_teacher.layers[rand_layer].neurons[rand_neuron].disregard[rand_disregard] = true;cout << "remove neuron in layer = " << rand_layer <<", neuron id ="<< rand_neuron <<" synapse id = "<< rand_disregard <<endl;

    }

    int disable_synapses(int max_inter, net::parameters param, net &the_teacher, Data &the_Data, int (net::*training_f)(Data*, net::parameters param))
    {
        vector<vector<bool> > damaged(the_teacher.size(), vector<bool>(max_inter, false));

        for(int i = 0;i<120;i++)
        {
            cout << "I " << i << endl;

            switch_off_synapse();

            assert(0);//check accuracy rate and test
            //the_teacher.test(&the_Data, "print result", 0.5);

            the_teacher.train(&the_Data, param, training_f);
        }

        return 1;
    }




    int print_gate(int t)
    {
        cout << indent(t);
        for(int i = 0, j = 0;i<the_Data.numInputs;i++)
        {
            if(j<gate.size())
            {
                if(gate[j].f.vector_id == i)
                {
                    if(gate[j].s == -1)
                    {
                        cout << "-";
                    }
                    else
                    {
                        assert(gate[j].s == 1);
                        cout << "+";
                    }
                    j++;
                }
                else
                {
                    cout << ".";
                }
            }
            else
            {
                cout << ".";
            }
        }
        if(gate.size() == 1)
        {
            bit_signature the_bit = gate[0].f;
            cout << " gate id: " << the_bit.vector_id;
            if(the_bit.num_operators == 2)
            {
                cout << " " << bitset<4>(the_bit.gate).to_string() << "(" << the_bit.operands[0] <<", "<< the_bit.operands[1] <<")";
            }
        }
        else
        {
            assert(gate.size() == 0);
        }
        cout << endl;
        bool enter = false;
        int ret = 1;
        if(new_left == NULL && new_right == NULL && left != NULL && right != NULL)
        {
            enter = true;
            ret+=left->print_gate(t+1);
            ret+=right->print_gate(t+1);
        }
        else if(new_left != NULL && new_right != NULL && left != NULL && right != NULL)
        {
            if(new_left->new_size + new_right->new_size > left->new_size+right->new_size)
            {
                //assert(0);
                enter = true;
                ret+=left->print_gate(t+1);
                ret+=right->print_gate(t+1);
            }
            else
            {
                enter = true;
                ret+=new_left->print_gate(t+1);
                ret+=new_right->print_gate(t+1);
            }
        }
        else
        {

            if(new_left != NULL && new_right != NULL && left == NULL && right == NULL)
            {
                enter = true;
                ret+=new_left->print_gate(t+1);
                ret+=new_right->print_gate(t+1);
            }
            else
            {
                assert(new_left == NULL && new_right == NULL && left == NULL && right == NULL);
            }
        }
        if(!enter)
        {
            for(int i = 0;i<the_Data.size();i++)
            {
                cout << indent(t) << the_Data.printInput(i) << " "<< the_Data.printOutput(i) << endl;
            }
        }
        cout << indent(t) << ret << " " << original_size+(!enter&((int)the_Data.size()<2)) << endl;
        return ret;
    }


    //int local_single_build(int _n, double min_rate, double max_rate , int parameter)
    int local_single_build(int _n, net::parameters param)
    {
        //string type = "input_is_output";
        string type = "longestSubstring_ai_is_1";
        //string type = "vector_loop_language";
        //string type = "a_i_is_a_i_plus_one_for_all";
        //string type = "longestSubstring_dai_di_is_0";

        int inputSize = _n, outputSize;

        the_Data.generateData(inputSize, outputSize, type);

        inputSize = the_Data.numInputs;
        outputSize = the_Data.numOutputs;
        int max_inter = inputSize;
        int hiddenLayers[10] =
                {
                        max_inter,
                        -1
                };

        the_teacher = net(inputSize, hiddenLayers, outputSize);
        return the_teacher.train(&the_Data, param, &net::softPriorityTrain);
        //assert(min_rate == max_rate);
        //return the_teacher.train(&the_Data, max_rate, &net::fullBatchTrain);
        //the_teacher.train(&the_Data, _rate, &net::hardPriorityTrain);
        //the_teacher.test(&the_Data, "print result");
        //the_teacher.analyzeMistakes(&the_Data);
    }
};

class neural_designed_circuit
{
public:


    /*int build_circuit_based_on_pairs(int _n, net::parameters param, int (net::*training_f)(Data*, net::parameters))
     {
     Data the_Data;
        //string type = "input_is_output";
        //string type = "longest_substring_double_to_middle";
        //string type = "longest_substring_of_two_strings";
        string type = "longestSubstring_ai_is_1";
        //string type = "vector_loop_language";
        //string type = "a_i_is_a_i_plus_one_for_all";
        //string type = "longestSubstring_dai_di_is_0";

        int init_inputSize = _n, outputSize;

        the_Data.generateData(init_inputSize, outputSize, type);

        while(1)
        {
            net the_teacher = net(the_Data.numInputs, the_Data.numOutputs);
            the_teacher.train(&the_Data, param, training_f);

            typedef net::data_model::bit_dimension_pair bit_dimension_pair;
            vector<bit_dimension_pair> sorted_pairs = the_teacher.the_model.sort_bit_dimension_pairs();

            int at = 0;
            bool enter = false;
            while(!enter)
            {
                bit_dimension_pair the_pair = sorted_pairs[at];
                cout << "Try :: :: " << the_pair.print() <<endl;
                operator_signature new_operator = operator_signature(the_pair.f_bit, the_pair.s_bit, the_pair.f_dim, the_pair.s_dim);

                Data data_with_best_pair;
                if(the_Data.apply_new_operator_to_data(new_operator, data_with_best_pair))
                {

                    net new_teacher = net(data_with_best_pair.numInputs, data_with_best_pair.numOutputs);
                    new_teacher.train(&data_with_best_pair, param, training_f);

                    int the_bit = new_teacher.the_model.get_worst_dimension();
                    cout << the_bit <<endl;
                    if(the_bit == data_with_best_pair.numInputs-1)
                    {
                        enter = true;
                        cout << "Finally :: :: " << the_pair.print() <<endl;
                        the_Data = data_with_best_pair;
                    }
                    else
                    {
                        cout << "Fail :: :: " << the_pair.print() <<endl;
                        at++;
                    }
                }
                else
                {
                    at++;
                }
            }
        }

    }*/

    int build_circuit_based_on_singletons(int _n, net::parameters param, int (net::*training_f)(Data*, net::parameters))
    {
        Data the_Data;
        //string type = "input_is_output";
        //string type = "longest_substring_double_to_middle";
        //string type = "longest_substring_of_two_strings";
        string type = "longestSubstring_ai_is_1";
        //string type = "vector_loop_language";
        //string type = "a_i_is_a_i_plus_one_for_all";
        //string type = "longestSubstring_dai_di_is_0";

        int inputSize = _n, outputSize;

        the_Data.generateData(inputSize, outputSize, type);

        inputSize = the_Data.numInputs;
        outputSize = the_Data.numOutputs;
        int max_inter = inputSize;
        int hiddenLayers[10] =
                {
                        max_inter,
                        -1
                };

        //cout << the_Data.size() <<endl;
        net the_teacher = net(inputSize, hiddenLayers, outputSize);
        the_teacher.train(&the_Data, param, training_f);

        //vector<bit_signature> bits_wanted = the_teacher.the_model.bit_weighted_negative_dimension;
        vector<bit_signature> bits_wanted = the_teacher.the_model.dimension_error_ratio_score;


        for(int i = 0;i<bits_wanted.size();i++)
        {
            bits_wanted[i].vector_id = i;
            bits_wanted[i].set_sort_by(abs(bits_wanted[i].value));
        }

        sort_v(bits_wanted);
        rev_v(bits_wanted);

        for(int i = 0;i<bits_wanted.size();i++)
        {
            cout << bits_wanted[i].value <<"(" << bits_wanted[i].vector_id <<"), ";
        }
        cout << endl;

        //vector<Data> all_Datas;
        //all_Datas.push_back(the_Data);

        //vector<vector<bit_signature> > all_bits_wanted;
        //all_bits_wanted.push_back(bits_wanted);

        priority_queue<pair<bit_signature, int> > best_pairs;

        bool printDecisionTree = true;

        int trials = 1;
        if(printDecisionTree)
        {
            cout << "decision tree with no circuit: " <<endl;
            for(int i = 0;i<trials;i++)
            {
                neural_decision_tree special_root;
                special_root.the_Data = the_Data;
                cout << special_root.build_tree(param, training_f) <<"\t";
                special_root.print_gate(0);
            }
            cout << endl;
        }

        for(int i = 0;i<bits_wanted.size();i++)
        {
            for(int j = i+1; j<bits_wanted.size();j++)
            {
                //cout << "insert " << bits_wanted[i].get_sort_by()+bits_wanted[j].get_sort_by() <<endl;
                bit_signature first_operand = bit_signature(bits_wanted[i].get_sort_by()+bits_wanted[j].get_sort_by(), i, j);
                best_pairs.push(mp(first_operand, 0));
            }
        }

        while(!best_pairs.empty())
        {
            pair<bit_signature, int> circuit_data = best_pairs.top();
            best_pairs.pop();

            //Data* the_local_Data = &all_Datas[circuit_data.s];
            Data* the_local_Data = &the_Data;

            bit_signature at_pair = circuit_data.f;

            //vector<bit_signature> local_bits_wanted = all_bits_wanted[circuit_data.s];
            vector<bit_signature> local_bits_wanted = bits_wanted;//all_bits_wanted[circuit_data.s];

            int loc_1 = at_pair.operands[0], loc_2 = at_pair.operands[1];
            operator_signature new_operator = operator_signature((local_bits_wanted[loc_1]>0), (local_bits_wanted[loc_2]>0), local_bits_wanted[loc_1].vector_id, local_bits_wanted[loc_2].vector_id);

            Data data_with_best_pair;

            if(the_local_Data->apply_new_operator_to_data(new_operator, data_with_best_pair))
            {
                net new_teacher = net(data_with_best_pair.numInputs, data_with_best_pair.numOutputs);
                new_teacher.train(&data_with_best_pair, param, training_f);

                vector<bit_signature> new_bits_wanted = new_teacher.the_model.dimension_error_ratio_score;

                for(int i = 0;i<new_bits_wanted.size();i++)
                {
                    new_bits_wanted[i].vector_id = i;
                    new_bits_wanted[i].set_sort_by(abs(new_bits_wanted[i].value));
                }

                sort_v(new_bits_wanted);
                rev_v(new_bits_wanted);

                /*cout << at_pair.get_sort_by() << " " << loc_1 <<" " << loc_2 << " :: ";
                for(int i = 0;i<new_bits_wanted.size();i++)
                {
                    cout << new_bits_wanted[i].value <<"(" << new_bits_wanted[i].vector_id <<"), " ;
                }
                cout << endl;*/

                //if(new_bits_wanted[0].vector_id == new_bits_wanted.size()-1)
                {
                    int base_id = 0;
                    cout << data_with_best_pair.numInputs-data_with_best_pair.circuit.size() <<" vars in " << data_with_best_pair.circuit.size() <<" gates :: ";
                    for(int i = 0;i<data_with_best_pair.circuit.size();i++)
                    {
                        assert(data_with_best_pair.circuit[i].operands.size() == 2);
                        cout << "gate(" << bitset<4>(data_with_best_pair.circuit[i].gate).to_string() <<", "
                             << data_with_best_pair.circuit[i].operands[0] <<", "<< data_with_best_pair.circuit[i].operands[1] <<"); ";
                    }
                    cout << endl;

                    if(printDecisionTree)
                    {
                        cout << "decision tree with this circuit: ";
                        for(int i = 0;i<trials;i++)
                        {
                            neural_decision_tree special_root;
                            special_root.the_Data = data_with_best_pair;
                            cout << special_root.build_tree(param, training_f) <<endl;
                            special_root.print_gate(0);
                        }
                        cout << endl;
                    }

                    //all_Datas.push_back(data_with_best_pair);
                    the_Data = data_with_best_pair;
                    //all_bits_wanted.push_back(new_bits_wanted);
                    bits_wanted = new_bits_wanted;

                    while(!best_pairs.empty())
                    {
                        best_pairs.pop();
                    }

                    for(int i = 0;i<new_bits_wanted.size();i++)
                    {
                        for(int j = i+1; j<new_bits_wanted.size();j++)
                        {
                            if(new_bits_wanted[i].get_sort_by() > 0 && new_bits_wanted[j].get_sort_by() > 0 )
                            {
                                bit_signature first_operand = bit_signature(new_bits_wanted[i].get_sort_by()+new_bits_wanted[j].get_sort_by(), i, j);
                                best_pairs.push(mp(first_operand, -1));
                            }
                        }
                    }
                }
            }
        }
        return 1;
    }

    void single_expansion_step(Data the_data, net::parameters param, vector<operator_signature> &operators, set<pair<int, pair<int, int> > > &dp_exists, bool &enter)
    {
        int num_in = the_data.numInputs, num_out = the_data.numOutputs;
        net first_teacher = net(num_in, num_out);
        first_teacher.train(&the_data, param, &net::softPriorityTrain);

        vector<vector<bit_signature> > sorted_iouts = first_teacher.the_model.sort_inout_dimensions();


        //vector<operator_signature> operators;

        for(int i = 0;i<num_out;i++)
        {
            vector<pair<double, pair<int, int> > > sorted_local_gates;
            for(int j = 0;j<num_in;j++)
            {
                for(int k = j+1;k<num_in;k++)
                {
                    sorted_local_gates.push_back
                            (mp(sorted_iouts[i][j].value+sorted_iouts[i][k].value, mp(sorted_iouts[i][j].vector_id, sorted_iouts[i][k].vector_id)));
                }
            }
            sort_v(sorted_local_gates);
            rev_v(sorted_local_gates);

            vector<pair<int, int> > new_gates;

            int width = 10;
            for(int j = 0;j<min((int)sorted_local_gates.size(), width);j++)
            {
                new_gates.push_back(sorted_local_gates[j].s);
            }

            Data local_data;
            //the_data.apply_dnf_important_pair_dimensions_to_data(new_gates, local_data);
            the_data.apply_dnf_important_pair_dimensions_to_data(new_gates, local_data);

            net second_teacher = net(local_data.numInputs, local_data.numOutputs);
            second_teacher.train(&local_data, param, &net::softPriorityTrain);

            vector<vector<bit_signature> > single_dimensions = second_teacher.the_model.sort_inout_dimensions();

            int top_cutoff = 1;
            for(int j = 0;j<min(top_cutoff, (int)single_dimensions[i].size());j++)
            {
                int potential_id = single_dimensions[i][j].vector_id;
                if(potential_id >= num_in)
                {
                    pair<int, pair<int, int> > new_gate = mp(local_data.circuit[potential_id].gate, local_data.circuit[potential_id].operands_to_pair());
                    if(dp_exists.find(new_gate) == dp_exists.end())
                    {
                        enter = true;
                        dp_exists.insert(new_gate);
                        operators.push_back(local_data.circuit[potential_id]);
                    }
                }
            }
        }

    }

    int build_circuit_per_output_dimension(Data the_data, net::parameters param)
    {
        set<pair<int, pair<int, int> > > dp_exists;
        bool enter = true;
        int new_dimension_init = 0;
        int new_dimension_end = the_data.numInputs-1;
        while(enter)
        {
            enter = false;
            vector<operator_signature> operators;
            for(int i = new_dimension_init; i<=new_dimension_end; i++)
            {
                operator_signature gate = the_data.circuit[i];
                Data left, right;
                the_data.split(gate, left, right);

                single_expansion_step(left, param, operators, dp_exists, enter);
                single_expansion_step(right, param, operators, dp_exists, enter);

            }

            Data augmented_data;
            the_data.apply_new_operators_to_data(operators, augmented_data);
            the_data = augmented_data;

            new_dimension_init = new_dimension_end+1;
            new_dimension_end = the_data.numInputs-1;

            for(int i = 0;i<the_data.circuit.size();i++)
            {
                cout << the_data.circuit[i].print() <<endl;
            }
            cout << endl;

            //the_data.printData("data");
            int score = the_data.get_score(true);
            cout << "score = " << score << endl;
        }
        return 1;
    }
};


void see_delta_w()
{
    const int n = 3;

    const int middle_layer = n;

    net walker = net(n, middle_layer, 1);

    vector<pair<int, int> > to_sort;

    vector<net> net_data;

    Data meta_task;

    net walker_init = net(n, middle_layer, 1);

    for(int i = 0; i<(1<<(1<<n))/8;i++)
    {
        Data the_data;
        the_data.init_exaustive_table_with_unary_output(n, i);

        net::parameters param = net::parameters(1, 1);//rate, batch_width

        walker.set_special_weights();
        //walker = walker_init;
        walker.save_weights();
        printOnlyBatch = false;
        cout << bitset<(1<<n)>(i).to_string()<<endl;
        param.track_dimension_model = false;
        param.accuracy = 0.25/2;
        to_sort.push_back(mp(walker.train(&the_data, param, &net::softPriorityTrain), i));

        net::test_score score = walker.test(&the_data, 0.25/2);

        assert(score.get_wrong() == 0);
        //walker.compare_to_save();
        //walker.printWeights();
        meta_task.push_back(to_bit_signature_vector((1<<n), i), walker.to_bit_signature_vector());
    }

    Data noramlized_meta_task;
    meta_task.normalize(noramlized_meta_task);

    /*for(int i = 0;i<meta_task.size();i++)
    {
        meta_task.printTest(i);
    }*/

    net net_learning_nets = net(noramlized_meta_task.numInputs, 2*noramlized_meta_task.numInputs, noramlized_meta_task.numOutputs);

    net::parameters param = net::parameters(1, 3);//rate, batch_width
    printItteration = true;
    param.track_dimension_model = false;
    param.accuracy = 0.0125*2;
    net_learning_nets.train(&noramlized_meta_task, param, &net::softPriorityTrain);

    Data resulting_data;

    int total_error = 0;
    for(int i = 0; i<noramlized_meta_task.size();i++)
    {
        vector<bit_signature> network_output = net_learning_nets.forwardPropagate(to_bit_signature_vector((1<<n), i), false);

        noramlized_meta_task.unnormalize(network_output);

        net output_network = net(n, middle_layer, 1, network_output);
        //output_network.printWeights();

        Data the_data;
        the_data.init_exaustive_table_with_unary_output(n, i);

        net::test_score score = output_network.test(&the_data, 0.5);
        for(int i = 0;i<score.correct_examples.size();i++)
        {
            cout << score.correct_examples[i];
        }
        total_error += score.get_wrong();
        cout << " #wrong = " << score.get_wrong() <<endl;
    }
    cout << total_error << " over " << noramlized_meta_task.size() << endl;

    /*
    for(int i = 0; i<(1<<(1<<n));i++)
    {
        Data the_data;
        the_data.init_exaustive_table_with_unary_output(n, i);

        net::parameters param = net::parameters(1, 1);//rate, batch_width

        walker.set_special_weights();
        //walker.save_weights();
        printOnlyBatch = true;
        cout << bitset<(1<<n)>(i).to_string()<<" ";
        to_sort.push_back(mp(walker.train(&the_data, param, &net::softPriorityTrain), i));
        //walker.compare_to_save();
    }
    cout << endl;
    sort_v(to_sort);
    for(int i = 0;i<to_sort.size();i++)
    {
        cout << bitset<(1<<n)>(to_sort[i].s).to_string() <<endl;
    }*/
}

class graph_plotter
{
public:
    void init()
    {
        printCycle = false;
        printItteration = false;
        printCleanItteration = false;

        print_delta_knowledge_graph = false;
        print_important_neurons = false;
        print_classify_neurons = false;
        print_implications = false;
        print_the_imporant_bit = false;

        print_discrete_model = false;
        printNewChildren = false;

        print_tree_synthesys = false;
        print_try_and_finally = false;
        printOnlyBatch = false;


        if(true)
        {

            if(false)
            {
                cout << "See delta_w" << endl;
                see_delta_w();
                return;
            }
            if(true)
            {
                latentDecisionTreeExtractor SpicyAmphibian = latentDecisionTreeExtractor();

                for(int ensamble_size = 1; ensamble_size <= 10 ; ensamble_size++)
                {
                        vector<int> leaf_iters;
                        leaf_iters.pb(-1);//no leaf_iter for n = 0;
                        leaf_iters.pb(4);// for n = 1;
                        leaf_iters.pb(16);// for n = 2;
                        leaf_iters.pb(24);// for n = 3;
                        leaf_iters.pb(32);// for n = 4;
                        leaf_iters.pb(64);// for n = 5;

                        vector<int> hidden_layer_width;
                        hidden_layer_width.pb(-1);//no leaf_iter for n = 0;
                        hidden_layer_width.pb(2);// for n = 1;
                        hidden_layer_width.pb(4);// for n = 2;
                        hidden_layer_width.pb(6);// for n = 3;
                        hidden_layer_width.pb(8);// for n = 4;
                        hidden_layer_width.pb(10);// for n = 5;


                        vector<int> hidden_layer_depth;
                        hidden_layer_depth.pb(-1);//no leaf_iter for n = 0;
                        hidden_layer_depth.pb(1);// for n = 1;
                        hidden_layer_depth.pb(1);// for n = 2;
                        hidden_layer_depth.pb(1);// for n = 3;
                        hidden_layer_depth.pb(1);// for n = 4;
                        hidden_layer_depth.pb(1);// for n = 5;

                        int min_trainable_n = 4;
                        int max_trainable_n = 4;

                        vector<firstOrderDatasets<DataAndScore> > datasets;
                        for(int i = 0;i<min_trainable_n-1;i++)
                        {
                            datasets.pb(firstOrderDatasets<DataAndScore>());//n < min_trainable_n
                        }

                        assert(min_trainable_n == max_trainable_n);

                        for(int i = min_trainable_n-1; i<max_trainable_n; i++)
                        {
                            //datasets.pb(init_smallest_train_set(max_trainable_n-1));//n = min_trainable_n
                            datasets.pb(init_custom_data_and_difficulty_set(i));
                        }

                        datasets.pb(init_random_train_set(max_trainable_n));

                        for(int repeat = 0; repeat < 10; repeat++) {
                            SpicyAmphibian.train_library<DataAndScore>
                            (min_trainable_n, max_trainable_n, leaf_iters, hidden_layer_width, hidden_layer_depth,
                                    ensamble_size, datasets);
                        }
                }
                return;
            }
            if(false)
            {
                dp_decision_tree<Data> dper;
                printItteration = false;
                print_close_local_data_model = false;
                dper.run();
                return;
            }

            if(false)
            {
                string type = "longestSubstring_ai_is_1";
                Data the_data = Data();
                int n = 6, outputSize;
                the_data.generateData(n, outputSize, type);

                printItteration = true;
                net::parameters param = net::parameters(1.5, 3);
                param.num_stale_iterations = 10000;//7;
                param.set_iteration_count(16);//(16);
                param.ensamble_size = 1;//8;
                param.priority_train_f = &net::softPriorityTrain;

                decision_tree tree = decision_tree(&the_data, param);

                return ;
            }

            if(false) //via circuit
            {
                net::parameters param = net::parameters(1.5, 3);

                neural_designed_circuit two_nets;
                param.num_stale_iterations = 7;
                param.set_iteration_count(16);

                string type = "longestSubstring_ai_is_1";
                Data the_data = Data();
                int n = 4, outputSize;
                the_data.generateData(n, outputSize, type);

                two_nets.build_circuit_per_output_dimension(the_data, param);
                //two_nets.build_circuit_based_on_singletons(6, param, &net::softPriorityTrain);
                return;
            }


            if(false)
            {
                //single init root/local build
                neural_decision_tree tree = neural_decision_tree();
                //tree.local_single_build(10, net::parameters(1, 2));

                //return ;

                net::parameters param = net::parameters(3.79, 3);
                //net::parameters param = net::parameters(1, 3);

                param.num_stale_iterations = 3;
                int to_print = tree.init_root(8, param, &net::softPriorityTrain);

                cout << to_print << endl;

                int num = tree.print_gate(0);

                cout << "num gates = " << num <<endl;

                cout << endl;
                return ;
            }



            print_discrete_model = false;
            printItteration = false;
            printNewChildren = false;

            //for(double id = 0.5;id<=10;id+=0.5)
            for(int id_iter_count = 16; id_iter_count <= 16; id_iter_count*=2)
            {
                cout <<endl;
                cout << id_iter_count <<" ::: " << endl;
                for(int id_ensamble_size = 1;id_ensamble_size<=12;id_ensamble_size++)
                {
                    cout << id_ensamble_size << "\t" <<" :: " << "\t";
                    for(int i = 0;i<6;i++)
                    {
                        net::parameters param = net::parameters(1.5, 3);
                        param.num_stale_iterations = 7;
                        param.set_iteration_count(id_iter_count);
                        param.ensamble_size = id_ensamble_size;


                        string type = "longestSubstring_ai_is_1";
                        Data the_data = Data();
                        int n = 6, outputSize;
                        the_data.generateData(n, outputSize, type);

                        decision_tree tree = decision_tree(&the_data, param);
                        cout << tree.size <<"\t";

                        //neural_decision_tree tree = neural_decision_tree();
                        //cout << tree.init_root(6, param, &net::softPriorityTrain) <<"\t";

                        //int num = tree.print_gate(0);

                        //cout << "num gates = " << num <<endl;
                        //cout << endl <<endl;

                    }
                    cout << endl;
                }
            }


            //rate plotter
            /*double greates_rate = 3;
             for(double id = greates_rate;id>=1;id-=0.2)
             {
             cout << id << "\t" <<" :: " << "\t";
             for(int i = 0;i<40;i++)
             {
             cout << tree.local_single_build(5, id, greates_rate, 4) <<"\t";
             }
             cout << endl;
             }*/

            return ;
        }
        else
        {
            /*print_the_imporant_bit = false;
             for(double learning_rate = 3; true; learning_rate*=0.8)
             {
             cout << learning_rate << "\t" << ":" << "\t";
             for(int num_trials = 0; num_trials < 7; num_trials++)
             {
             neural_decision_tree tree = neural_decision_tree();
             int num_nodes = tree.init_root(8, learning_rate, 1, &net::queueTrain);
             cout  << num_nodes << "\t";
             }
             cout << endl;
             }*/
        }
    }
};

int main()
{
    //srand(time(0));
    clock_t count = clock();
    graph_plotter worker;
    worker.init();
    cout << "time elapsed = " << (double)(clock()-count)/CLOCKS_PER_SEC<<endl;
    return 0;
}



/*
 interesting thigs to try::

 fragment code into files
 stoping learning once a good dimension to cut is determiend
 do time and iteration tests based on learning rate, learning iteration termination, queue vs priority, different topologiez
 learning in batches of different size
 learning in weighted batcehs
 learning with gradients of learning rates


 John conway's learning
 Create iterativelly small machines. Have an algorithm that adjusts network topology.
 Select them based on some heuristic to do with learning them. For eg. which have highest gradients of decision tree size based on learning.



 */

