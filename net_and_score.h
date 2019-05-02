//
// Created by Kliment Serafimov on 2019-04-29.
//

#ifndef NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_NET_AND_SCORE_H
#define NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_NET_AND_SCORE_H

#include "Header.h"
#include "net.h"

class net_and_score: public net
{
public:

    double max_error = 0;
    double sum_error = 0;

    bool is_init_score = true;

    int num_train_fail = 0;

    int max_train_iter = 0;
    int sum_train_iter = 0;

    int max_leaf_iter = (1<<30);

    vector<double> individual_max_errors;

    int ordering_error = (1<<30);

    void update(net_and_score better)
    {

        max_leaf_iter = min(better.max_leaf_iter, max_leaf_iter);

        ordering_error = min(better.ordering_error, ordering_error);

        max_error = min(better.max_error, max_error);

    }

    operator double() const
    {
        return max_error;
    }

    void set_individual_scores(vector<net_and_score> individual_scores)
    {
        assert(individual_max_errors.size() == 0);
        for(int i = 0;i<individual_scores.size();i++)
        {
            individual_max_errors.pb(individual_scores[i].max_error);
        }
    }

    net_and_score()
    {

    }


    net_and_score(net self): net(self)
    {

    }

    void clear_vals()
    {
        max_error = 0;
        sum_error = 0;


        is_init_score = true;

        num_train_fail = 0;

        max_train_iter = 0;
        sum_train_iter = 0;

        max_leaf_iter = (1<<30);

        individual_max_errors.clear();

        ordering_error = (1<<30);
    }

    void update_max_leaf_iter(int new_iter)
    {
        if(max_leaf_iter == (1<<30))
        {
            max_leaf_iter = new_iter;
        }
        else{
            max_leaf_iter = max(max_leaf_iter, new_iter);
        }
    }

    bool operator < (net_and_score &other) const
    {
        if(is_init_score)
        {
            return false;
        }

        if(other.is_init_score)
        {
            return true;
        }

        return (max_error < other.max_error || max_leaf_iter < other.max_leaf_iter || ordering_error < other.ordering_error);


    }

    bool has_value()
    {
        return true;
    }

    string print()
    {
        return "max_error = \t" + std::to_string(max_error) + "\t max_iter = " + std::to_string(max_leaf_iter) +
        "\t ordering_error = " + std::to_string(ordering_error); // + "\t sum_error = \t" + std::to_string(sum_error);
    }

    string clean_print()
    {
        return std::to_string(max_error);// + "\t\t" + std::to_string(sum_error);
    }
};

class meta_net_and_score: public net_and_score
{
public:

    meta_net_and_score(net_and_score self): net_and_score(self)
    {

    }

    meta_net_and_score()
    {

    }

};

#endif //NEURAL_GUIDED_DECISION_TREE_SYNTHESIS_NET_AND_SCORE_H
