/**
 * @file cgs_lda.cc
 * @author  Shanshan Wang, Xiaoyang Han, Qiancheng Wei
 * @version 0.2
 *
 * @section LICENSE 
 * 
 * Copyright 2018 Shanshan Wang(wangshanshan171@ucas.ac.cn)

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * @section DESCRIPTION
 * 
 * This file implements the Collapsed Gibbs Sampler (CGS) for the Latent 
 * Dirichlet Allocation (LDA) model using graphlite API.
 *
 */


#include <iostream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <assert.h>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/atomic.hpp>
#include <boost/random.hpp>
#include <random>
#include <unistd.h>
#include <vector>

#include "GraphLite.h"

#define DEBUG // run on vm


/**
* \brief Change VERTEX_CLASS_NAME(name) definition to use a different class name 
*/
#define VERTEX_CLASS_NAME(name) cgs_lda##name

#define EPS 1e-6
/**
 * \brief Define the total number of topics 
 * If we use a single machine with 8 worker, we can only set NTOPICS as 10
 */
#define NTOPICS size_t(10)

/**
 * \brief number of docs, words, vertices
 *
 */
unsigned long long NDOCS;
unsigned long long NWORDS;
unsigned long long NVERTICES;

/**
 * \brief Define the flag of word or doc vertex
 */
#define IS_WORD -1
#define IS_DOC 1
#define IS_NULL 0


/**
 * \brief The alpha parameter determines the sparsity of topics for each document.
 */
double ALPHA = 1.0;

/**
 * \brief The Beta parameter determines the sparsity of words in each document.
 */
double BETA = 0.1;


/**
 * \brief We need a null topic to represent the topic assignment for tokens 
 * that have not yet been assigned.
 */
#define NULL_TOPIC long(-1)

/**
* \brief The vector type is used on each edge to store the
* assignments of each token.  There can be several occurrences of the
* same word in a given document and so a vector is used to store the
* assignments of each occurrence. It is used in edge_data.
* The vector type is also used to store the counts of tokens in
* each topic for words, documents, and assignments on vertices.
* It is used in vertex_data.
*/
typedef std::vector<long> vector_type;


/**
* \brief The vertex data represents each word and doc in the corpus and contains 
* the counts of tokens(word,doc pair) in each topic. 
* change vertex_data from struct to typedef struct for sume bug of graphlite.
*/
typedef struct vertexData{
    // The count of tokens in each topic.
    vector_type factor; 
    // judge vertex is word or doc
    int flag; 
    // number of outdegree for debug
    int64_t outdegree; 

}vertex_data;


/**
* \brief The edge data represents individual tokens (word,doc) pairs and their assignment to topics.
*/
typedef struct edgeData{
    // occurence of a word in a doc
    size_t ntoken;
    // The assignment of all tokens
    vector_type assignment;
}edge_data;

/**
* \brief The messageData is contains an array and a number, the array can't be defined as a vector.
*/
typedef struct messageData{
    // a vector of count
    long factor[NTOPICS];
    unsigned long long vid;//Vertex ID
}message_type;

/**
 * \brief global topic count is atomic, because it can be read and write for all vertices in the same worker.
 */
using boost::atomic;
boost::atomic<long> global_topic_count[NTOPICS] = {};

/**
 * \brief Data type of aggregator.
 */
typedef struct aggregator_struct{
    // global topic count
    long count[NTOPICS];
    double lik_words_given_topics;
    double lik_topics;
    // likelihood
    double likelihood;
}aggr_type;

/**
 * Computing log_gamma can be a bit slow so this class precomptues
 * log gamma for a subset of values.
 */
class log_gamma {
  double offset;
  std::vector<double> values;
public:
  log_gamma(): offset(1.0) {}

  void init(const double& new_offset, const size_t& buckets) {
    using boost::math::lgamma;
    assert(offset > 0.0);
    values.resize(buckets);
    offset = new_offset;
    for(size_t i = 0; i < values.size(); ++i) {
      values[i] = lgamma(i + offset);
    }
  }

  double operator()(const long& index) const {
    using boost::math::lgamma;
    if(index < values.size() && index >= 0) { return values[index]; }
    else { return lgamma(index + offset); }
  }

};

/**
 * \brief log gamma for alpha and beta. Attention that these should be 
 * initialized on load graph.
 */
log_gamma ALPHA_LGAMMA;
log_gamma BETA_LGAMMA;

/**
 * \brief VERTEX_CLASS_NAME(InputFormatter) can be kept as is
 */
/**
* \brief Our graph is in this format:
* 1st row: m_total_vertex NDOCS NWORDS
* 2nd row: m_total_edge
* other row: vid vid weight.ntoken
*/
class VERTEX_CLASS_NAME(InputFormatter): public InputFormatter {
public:
    int64_t getVertexNum() {
        unsigned long long n;
        sscanf(m_ptotal_vertex_line, "%lld %lld %lld", &n, &NDOCS, &NWORDS);
        NVERTICES = NDOCS+NWORDS;
        m_total_vertex= n;
        return m_total_vertex;
    }
    int64_t getEdgeNum() {
        unsigned long long n;
        sscanf(m_ptotal_edge_line, "%lld", &n);
        m_total_edge= n;
        return m_total_edge;
    }
    int getVertexValueSize() {
        // add vertex data type
        m_n_value_size = sizeof(vertex_data);
        return m_n_value_size;
    }
    int getEdgeValueSize() {
        m_e_value_size = sizeof(edge_data);
        return m_e_value_size;
    }
    int getMessageValueSize() {
        m_m_value_size = sizeof(message_type);
        return m_m_value_size;
    }
    void loadGraph() {
        unsigned long long last_vertex;
        unsigned long long from;
        unsigned long long to;
        edge_data weight;

        vertex_data value;

        int outdegree = 0;

        const char *line= getEdgeLine();

        // read edge weight as edge_data
        sscanf(line, "%lld %lld %zu", &from, &to, &(weight.ntoken));

        addEdge(from, to, &weight);

        last_vertex = from;
        ++outdegree;
        for (int64_t i = 1; i < m_total_edge; ++i) {
            line= getEdgeLine();

            // read edge weight as edge_data
            sscanf(line, "%lld %lld %zu", &from, &to, &(weight.ntoken));

            if (last_vertex != from) {
                addVertex(last_vertex, &value, outdegree);
                last_vertex = from;
                outdegree = 1;
            } else {
                ++outdegree;
            }
            addEdge(from, to, &weight);
        }
        addVertex(last_vertex, &value, outdegree);

    }
    // initialize log gamma of alpha and beta
    void init_LGAMMA(){
        ALPHA_LGAMMA.init(ALPHA, 100000);
        BETA_LGAMMA.init(BETA, 100000);
    }


};
/** 
 * \brief VERTEX_CLASS_NAME(OutputFormatter): his is where the output is generated .
 */
/**
 * \brief Output of docs and words is different. Output doc-topic matrix or word-topic matrix.
 */
class VERTEX_CLASS_NAME(OutputFormatter): public OutputFormatter {
public:
    void writeResult() {
        int64_t vid;
        vertex_data value;
        char s[1024];

        for (ResultIterator r_iter; ! r_iter.done(); r_iter.next() ) {
            r_iter.getIdValue(vid, &value);
            int n;

            std::string factor = "";
            for(size_t t = 0; t < NTOPICS; t++){
                 char temp[20];
                 sprintf(temp, "%-10lld", (unsigned long long)(value.factor[t]));
                 factor += temp;
            }
            if(value.flag == IS_DOC){
                n = sprintf(s, "%-5s:%-10lld: %s\n", "doc", (unsigned long long)vid, factor.c_str());
            }else{
                int64_t id = std::max(vid - int64_t(NDOCS), int64_t(0));
                n = sprintf(s, "%-5s:%-10lld: %s\n", "word", (unsigned long long)id, factor.c_str());
            }
            writeNextResLine(s, n);
        }
    }
};




/**
 * \brief VERTEX_CLASS_NAME(Aggregator): you can implement other types of aggregation 
 * the <type> is the type name of aggregator result value
 */
/**
 * \brief Aggregator of global_topic_count and likelihood
 */
class VERTEX_CLASS_NAME(Aggregator): public Aggregator<aggr_type> {
public:
    // type of m_global and m_local is the same with <type name>
    void init() {
        m_global.lik_words_given_topics = 0.0;
        m_global.lik_topics = 0.0;
        m_global.likelihood = 0.0;
        m_local.lik_words_given_topics = 0.0;
        m_local.lik_topics = 0.0;
        m_local.likelihood = 0.0;
    }

    void* getGlobal() {
        return &m_global;
    }

    // type of p is <type name>
    void setGlobal(const void* p) {
        using boost::math::lgamma;
        aggr_type* aggr = (aggr_type *) p;

        //global_topic_count
        for(int t = 0; t < NTOPICS; t++){
             m_global.count[t] = aggr->count[t]/2;
             global_topic_count[t] = m_global.count[t];
        }

        //likelihood
        double denominator = 0.0; // the denominator of the formula in paper
        for(size_t t = 0; t < NTOPICS; ++t) {
            const long value = std::max(long(m_global.count[t]), long(0));
            denominator += lgamma(value + NWORDS * BETA);
        } 

        m_global.lik_words_given_topics = NTOPICS * (lgamma(NWORDS * BETA) - NWORDS * lgamma(BETA)) -denominator + aggr->lik_words_given_topics;

        m_global.lik_topics = NDOCS * (lgamma(NTOPICS * ALPHA) - NTOPICS * lgamma(ALPHA)) + aggr->lik_topics;

        m_global.likelihood = m_global.lik_words_given_topics + m_global.lik_topics;
    }

    void* getLocal() {
        return &m_local;
    }

    // type of p is <type name>
    void merge(const void* p) {
        aggr_type* aggr = (aggr_type *) p;
        //global_count_topic
        for(int t = 0; t < NTOPICS; t++){
             m_global.count[t] += aggr->count[t];
        }
        //likelihood
        m_global.lik_words_given_topics += aggr->lik_words_given_topics;
        m_global.lik_topics += aggr->lik_topics;
    }

    // type of p is the type of value in AccumulateAggr(0, &value)
    void accumulate(const void* p) {
        using boost::math::lgamma;
        vertex_data val = *(vertex_data*) p;
        vector_type& factor = val.factor;  
        // global_count_topic
        for(int t = 0; t < NTOPICS; t++){
              m_local.count[t] += factor[t];
        }
        // likelihood
        int flag = val.flag;
        double lik_words_given_topics = 0.0;
        double lik_topics = 0.0;
        if(flag==IS_WORD){
            for(size_t t = 0; t < NTOPICS; ++t) {
                const long value = std::max(long(factor[t]), long(0));
                lik_words_given_topics += BETA_LGAMMA(value);
            }
        }else{
            double ntokens_in_doc = 0;
            for(size_t t = 0; t < NTOPICS; ++t) {
                const long value = std::max(long(factor[t]), long(0));
                lik_topics += ALPHA_LGAMMA(value);
                ntokens_in_doc += value;
            }
            lik_topics -= lgamma(ntokens_in_doc + NTOPICS * ALPHA);
        }
        m_local.lik_words_given_topics += lik_words_given_topics;
        m_local.lik_topics += lik_topics;

    }
};


/** VERTEX_CLASS_NAME(): the main vertex program with compute() */
/**
 * \brief The gibbs sampling in supersteps.
 *  Each two supersteps are used as one Gibbs sample.
 *  
 * superstep 0 : initialize variables
 * superstep even : word vertex sample, update self, assignment edges, send to doc vertex 
 * superstep odd : doc vertex receive from word vertex, update self, send to word vetex
 * 
 */
class VERTEX_CLASS_NAME(): public Vertex <vertex_data, edge_data, message_type> {
public:
    void compute(MessageIterator* pmsgs) {

        vertex_data val;

        // superstep 0
        if(getSuperstep() == 0){

            //  initialize variables
            val.factor.assign(NTOPICS, 0);
            val.outdegree = get_outdegree();
            if(getVertexId() < NDOCS) val.flag = 1;
            else val.flag = -1;
            assert(getVertexId() < NVERTICES);

            // word vertex assign topic to edge and send edge data to to doc
            if(val.flag == IS_WORD){

                // initialize edge assignment to NULL_TOPIC
                init_edge_topic();

                vector_type& word_topic_count = val.factor;

                //iterate all outedges and compute assignment topic
                OutEdgeIterator out_edge_it = getOutEdgeIterator();
                for ( ; !out_edge_it.done(); out_edge_it.next()){

                    // probablity of multimonial
                    std::vector<double> prob(NTOPICS);

                    // update in sampling and send to doc vertex
                    vector_type doc_topic_count(NTOPICS, 0);

                    // temp variable assignment, need to be passed to edge data after sampling
                    vector_type assignment = out_edge_it.getValue().assignment;

                    // sample and compute each assigment on one outedge
                    for(size_t t = 0; t < assignment.size(); t++){

                        // asg is a referenc of assignment[t]
                        long& asg = assignment[t];

		                // compute probability of multinomial
		                for(size_t t = 0; t < NTOPICS; ++t){

		                    const double n_dt = std::max(long(doc_topic_count[t]), long(0));
		                    const double n_wt = std::max(long(word_topic_count[t]), long(0));
		                    const double n_t  = std::max(long(global_topic_count[t]), long(0));
		                    prob[t] = (ALPHA + n_dt) * (BETA + n_wt) / (BETA * NWORDS + n_t);
                            
		  		        }

                        // get new asg using random multinomial
		                asg = multinomial(prob);

                        // update topic count of doc, word and global
		                ++doc_topic_count[asg];
		  			    ++word_topic_count[asg];
		  			    ++global_topic_count[asg];
		            }

                    // pass new assignment to current outedge data
                    Edge* edge = (Edge *)(out_edge_it.current());
                    edge_data* p = (edge_data *)(edge->weight);
                    for(size_t t = 0; t < assignment.size(); t++)
                        (p->assignment)[t] = assignment[t];

                    // send new assignment to doc vertex
                    unsigned long long vid_to = out_edge_it.target();
	                message_type ms_send;
		            ms_send.vid = getVertexId();
		            for(size_t t = 0; t < NTOPICS;t++){
                        ms_send.factor[t] = doc_topic_count[t];
                    }
                    sendMessageTo(vid_to, ms_send);


                }
                   
            }

        }else{// superstep != 0

            // get old vertex value
            val = getValue();

            // superstep odd: doc vertex work
		    if(getSuperstep() % 2 == 1){
       
                // doc receive, update and send to word
                if(val.flag==IS_DOC){
 
                    //receive new assignment from word vertex
                    for (;!pmsgs->done(); pmsgs->next()){	

                        // update count of topics
                        for(size_t t = 0 ; t < NTOPICS; t++){
			     			val.factor[t] += pmsgs->getValue().factor[t];
                        }
                    }  

                    // send doc vertex factor to all neighbor word vertex
                    message_type ms_send;
                    ms_send.vid = getVertexId();
                    for(size_t t = 0; t < NTOPICS;t++){
                        ms_send.factor[t] = val.factor[t];
                    }
                    sendMessageToAllNeighbors(ms_send);
                }

                // compute aggregator
	            accumulateAggr(0, &val);
		    }

            // superstep even : word vertex work
		    if(getSuperstep() % 2 == 0){

                // check aggregator result and judge if votetohalt
				aggr_type* aggr = (aggr_type *)getAggrGlobal(0);
                double likelihood = aggr->likelihood;
                if(int64_t(getVertexId())==int64_t(0)){
                    printf("lik_words_given_topics = %e\n", aggr->lik_words_given_topics);
                    printf("lik_topics = %e\n", aggr->lik_topics);
                    printf("likelihood = %e\n", aggr->likelihood);
                }


               // vote to halt
			    if(getSuperstep() > 10){// if likelihood < ESP
			        voteToHalt(); return;   
			    }

                // word receive, update, assignment and send 
                if(val.flag == IS_WORD){

                    vector_type& word_topic_count = val.factor;

                    // count of msg should equal to count of 
                    long count_msg = 0;

                    // each msg from one doc
                    // update the edge which connect current doc and this word
                    for (;!pmsgs->done();pmsgs->next()){

                         unsigned long long vid_from = pmsgs->getValue().vid;
                         vector_type doc_topic_count(NTOPICS, 0);
                         vector_type doc_topic_change(NTOPICS, 0);
                         for(size_t t; t < NTOPICS; t++){
				             doc_topic_count[t] = pmsgs->getValue().factor[t];
                         }
                         
                         OutEdgeIterator out_edge_it = getOutEdgeIterator();
                         //iterate outedges to find the edge connect to current doc
                         for (; !out_edge_it.done(); out_edge_it.next()){

                              unsigned long long vid_to = out_edge_it.target();

                              // if the edge connects to current doc, update the assignment on this edge    
                              // and send message to current doc and break for
                              if(vid_from == vid_to){

                                count_msg++;      
                          
                                // use pointer p to change edge data.
                                Edge* edge = (Edge *)(out_edge_it.current());
                                edge_data* p = (edge_data *)(edge->weight);
                                vector_type assignment = p->assignment;

                                // probability of multinomial
                                std::vector<double> prob(NTOPICS);

                                // sample and compute each new assignment 
						        for(size_t t = 0; t < assignment.size(); t++){

						            // asg is a referenc of assignment[t]
						            long& asg = assignment[t];

		                            --doc_topic_count[asg];
                                    --doc_topic_change[asg];
   			                        --word_topic_count[asg];
    		                        --global_topic_count[asg];
                                  
								    // compute probability of multinomial
								    for(size_t t = 0; t < NTOPICS; ++t){
								        const double n_dt = std::max(long(doc_topic_count[t]), long(0));
								        const double n_wt = std::max(long(word_topic_count[t]), long(0));
								        const double n_t  = std::max(long(global_topic_count[t]), long(0));
								        prob[t] = (ALPHA + n_dt) * (BETA + n_wt) / (BETA * NWORDS + n_t);
					  		        }

						            // get new asg using random multinomial
								    asg = multinomial(prob);

                                    // update topic counts of doc, word and global
								    ++doc_topic_count[asg];
                                    ++doc_topic_change[asg];
					  			    ++word_topic_count[asg];
					  			    ++global_topic_count[asg];


								}

						        // update assignment on current outedge data
						        for(size_t t = 0; t < assignment.size(); t++)
						            (p->assignment)[t] = assignment[t];

						        // send new assignment to doc vertex
							    message_type ms_send;
								ms_send.vid = getVertexId();
								for(size_t t = 0; t < NTOPICS;t++){
						            ms_send.factor[t] = doc_topic_change[t];
						        }
						        sendMessageTo(vid_to, ms_send);

                                break; // end of outedge iterator
                              }
                         }
                         
                    }
                    assert( count_msg == val.outdegree);
                }
		    }
        }

        // update vertex data
	    * mutableValue() = val;

    }

    // get outdegree of this vertex
    int64_t get_outdegree(){
        int64_t rt = getOutEdgeIterator().size();
        return rt;
    }

    // initialize topic on all edges as NULL_TOPIC
    void init_edge_topic(){
        OutEdgeIterator outEdges = getOutEdgeIterator();
   
        for ( ; ! outEdges.done(); outEdges.next() ) {

            Edge* edge = (Edge *)(outEdges.current());
            edge_data* p = (edge_data *)(edge->weight);
            p->assignment.assign(p->ntoken, NULL_TOPIC);
        }
        return ;
    }


    // copy from graphlab, generate a random number from a multinomial
    size_t multinomial(const std::vector<double>& prb) 
    {
		std::default_random_engine generator;
	  	std::uniform_real_distribution<double> distribution(0.0,1.0);
		assert(prb.size()>0);
		if (prb.size() == 1) return 0;
		double sum(0);
		for(size_t i = 0; i < prb.size(); ++i){
			assert(prb[i]>=0); 
			sum += prb[i];
		}
		assert(sum>0);
		const double rnd(distribution(generator));

		size_t ind = 0;
		for(double cumsum(prb[ind]/sum); rnd > cumsum && (ind+1) < prb.size(); cumsum += (prb[++ind]/sum));
		return ind;
    }

};

/** 
 * \brief VERTEX_CLASS_NAME(Graph): set the running configuration here.
 * we use 1 master and 8 workers.
 */
class VERTEX_CLASS_NAME(Graph): public Graph {
public:
    VERTEX_CLASS_NAME(Aggregator)* aggregator;

public:
    // argv[0]: cgs_lda.so
    // argv[1]: <input path>
    // argv[2]: <output path>
    void init(int argc, char* argv[]) {
/*
        setNumHosts(5);
        setHost(0, "localhost", 1411);
        setHost(1, "localhost", 1421);
        setHost(2, "localhost", 1431);
        setHost(3, "localhost", 1441);
        setHost(4, "localhost", 1451);
*/

        int nworkers = 9;
        setNumHosts(nworkers);
        for(int i = 0; i < nworkers; i++){
           setHost(i, "localhost", 1411 + i*10);
        }


        if (argc < 3) {
           printf ("Usage: %s <input path> <output path>\n", argv[0]);
           exit(1);
        }

        m_pin_path = argv[1];
        m_pout_path = argv[2];

        aggregator = new VERTEX_CLASS_NAME(Aggregator)[1];
        regNumAggr(1);
        regAggr(0, &aggregator[0]);

    }

    void term() {
        delete[] aggregator;
    }
};

/* 
 * \brief STOP: do not change the code below. 
 */
extern "C" Graph* create_graph() {
    Graph* pgraph = new VERTEX_CLASS_NAME(Graph);

    pgraph->m_pin_formatter = new VERTEX_CLASS_NAME(InputFormatter);
    pgraph->m_pout_formatter = new VERTEX_CLASS_NAME(OutputFormatter);
    pgraph->m_pver_base = new VERTEX_CLASS_NAME();

    return pgraph;
}

extern "C" void destroy_graph(Graph* pobject) {
    delete ( VERTEX_CLASS_NAME()* )(pobject->m_pver_base);
    delete ( VERTEX_CLASS_NAME(OutputFormatter)* )(pobject->m_pout_formatter);
    delete ( VERTEX_CLASS_NAME(InputFormatter)* )(pobject->m_pin_formatter);
    delete ( VERTEX_CLASS_NAME(Graph)* )pobject;
}
