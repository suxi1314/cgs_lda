/**
 * @file cgs_lda.cc
 * @author  Shanshan Wang
 * @version 0.1
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
#include <random>
#include <unistd.h>

#include "GraphLite.h"

#define DEBUG // run on vm


/**
* change VERTEX_CLASS_NAME(name) definition to use a different class name 
*/
#define VERTEX_CLASS_NAME(name) cgs_lda##name

#define EPS 1e-6
/**
 * the total number of topics to uses
 * the single machine with 8 worker can only set NTOPICS as 10
 */
#define NTOPICS size_t(10)

/**
 * the alpha parameter determines the sparsity of topics for each document.
 */
double ALPHA = 1;

/**
 * the Beta parameter determines the sparsity of words in each document.
 */
double BETA = 0.1;

/**
 * We use the factor type in aggregator, so we define an operator+=
 */

// We require a null topic to represent the topic assignment for tokens that have not yet been assigned.
#define NULL_TOPIC long(-1)

/**
* The assignment type is used on each edge to store the
* assignments of each token.  There can be several occurrences of the
* same word in a given document and so a vector is used to store the
* assignments of each occurrence.
*/

#define IS_WORD -1
#define IS_DOC 1
#define IS_NULL 0


typedef std::vector<long> vector_type;


/**
* The vertex data represents each word and doc in the corpus and contains 
* the counts of tokens(word,doc pair) in each topic. 
* change vertex_data from struct to typedef struct for sume bug of graphlite.
*/
typedef struct vertexData{
    // The count of tokens in each topic.
    vector_type factor;
    int flag;
    int64_t outdegree;

}vertex_data;


unsigned long long NDOCS;
unsigned long long NWORDS;
unsigned long long NVERTICES;

/**
* The edge data represents individual tokens (word,doc) pairs and their assignment to topics.
*/
typedef struct edgeData{
    size_t ntoken;
    // The assignment of all tokens
    vector_type assignment;
}edge_data;


typedef struct
{
    //int a;
    long factor[NTOPICS];
    unsigned long long vid;//Vertex ID
}message_type;

/** VERTEX_CLASS_NAME(InputFormatter) can be kept as is */
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

        // Note: modify this if an edge weight is to be read
        //       modify the 'weight' variable

        // read edge weight
        sscanf(line, "%lld %lld %zu", &from, &to, &(weight.ntoken));

        addEdge(from, to, &weight);

        last_vertex = from;
        ++outdegree;
        for (int64_t i = 1; i < m_total_edge; ++i) {
            line= getEdgeLine();

            // Note: modify this if an edge weight is to be read
            //       modify the 'weight' variable

            // read edge weight
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
};
/** VERTEX_CLASS_NAME(OutputFormatter): 
his is where the output is generated */
class VERTEX_CLASS_NAME(OutputFormatter): public OutputFormatter {
public:
    void writeResult() {
        int64_t vid;
        vertex_data value;
        char s[1024];

        for (ResultIterator r_iter; ! r_iter.done(); r_iter.next() ) {
            r_iter.getIdValue(vid, &value);
            int n = sprintf(s, "%lld:%d:%lld:%ld\n", (unsigned long long)vid, value.flag, (unsigned long long)value.outdegree, value.factor[0]);
            writeNextResLine(s, n);
        }
    }
};

typedef struct aggregator_struct{
    long count[NTOPICS];
    double lik_words_given_topics = 0.0;
    double lik_topics = 0.0;
    double likelihood = 0.0;
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

log_gamma ALPHA_LGAMMA;
log_gamma BETA_LGAMMA;

/** VERTEX_CLASS_NAME(Aggregator): you can implement other types of aggregation */
// the <type name> is the type of aggregator result value
class VERTEX_CLASS_NAME(Aggregator): public Aggregator<aggr_type> {
public:
    // type of m_global and m_local is the same with <type name>
    void init() {
        m_global;
        m_local;
    }

    void* getGlobal() {
        return &m_global;
    }

    // type of p is <type name>
    void setGlobal(const void* p) {
        using boost::math::lgamma;
        aggr_type aggr = *(aggr_type *) p;
        m_global.count[0] = aggr.count[0];
        //global_topic_count
        for(int t = 0; t < NTOPICS; t++) m_global.count[t] = aggr.count[t]/2;
        //likelihood
        double denominator = 0; // the denominator of the formula
        for(size_t t = 0; t < NTOPICS; ++t) {
            denominator += lgamma(long(m_global.count) + NWORDS * BETA);
        } 
        m_global.lik_words_given_topics = NTOPICS * (lgamma(NWORDS * BETA) 
                                            - NWORDS * lgamma(BETA)) 
                                            - denominator 
                                            + aggr.lik_words_given_topics;
        m_global.lik_topics = NDOCS * (lgamma(NTOPICS * ALPHA) 
                            - NTOPICS * lgamma(ALPHA)) 
                            + aggr.lik_topics;
        m_global.likelihood = m_global.lik_words_given_topics + m_global.lik_topics;
    }

    void* getLocal() {
        return &m_local;
    }

    // type of p is <type name>
    void merge(const void* p) {
        aggr_type aggr = *(aggr_type *) p;
        //global_count_topic
        for(int t = 0; t < NTOPICS; t++){
             m_global.count[t] += aggr.count[t];
        }
        //likelihood
        m_global.lik_words_given_topics += aggr.lik_words_given_topics;
        m_global.lik_topics += aggr.lik_topics;
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
                lik_words_given_topics += BETA_LGAMMA(long(factor[t]));
                // printf("%ld %lf\n", long(factor[t]), lik_words_given_topics);
            }
        }else{
            double ntokens_in_doc = 0;
            for(size_t t = 0; t < NTOPICS; ++t) {
                lik_topics += ALPHA_LGAMMA(long(factor[t]));
                ntokens_in_doc += long(factor[t]);
            }
            lik_topics -= lgamma(ntokens_in_doc + NTOPICS * ALPHA);
        }
        m_local.lik_words_given_topics += lik_words_given_topics;
        m_local.lik_topics += lik_topics;

    }

};


/** VERTEX_CLASS_NAME(): the main vertex program with compute() */
class VERTEX_CLASS_NAME(): public Vertex <vertex_data, edge_data, message_type> {
public:
    void compute(MessageIterator* pmsgs) {

        vertex_data val;
	    vector_type ass_send, ass_recv;

        // superstep 0 initialize variables
        if(getSuperstep() == 0){
            val.factor.assign(NTOPICS, 0);
            val.outdegree = get_outdegree();
            if(getVertexId() < NDOCS) val.flag = 1;
            else val.flag = -1;
            assert(getVertexId() < NVERTICES);

            // if current vertex is word, assign topic to edge
            // and send edge data to to doc
            if(val.flag == IS_WORD){
                // initialize edge assignment to NULL_TOPIC
                init_edge_topic();

                std::vector<double> prob(NTOPICS);
                long global_topic_count[NTOPICS];
                long doc_topic_count[NTOPICS];
                long word_topic_count[NTOPICS];

                for(size_t t = 0; t < NTOPICS; t++){
                    global_topic_count[t] = 0;
                    word_topic_count[t] = 0;
                    doc_topic_count[t] = 0;
                }
                OutEdgeIterator out_edge_it = getOutEdgeIterator();
                //iterate all outedges and compute assignment topic
                for ( ; !out_edge_it.done(); out_edge_it.next()){
                    // temp variable assignment
                    // need to be passed to edge data
                    vector_type assignment = out_edge_it.getValue().assignment;
                    // compute assigment of one outedge
                    for(size_t t = 0; t < assignment.size(); t++){
                        // asg is a referenc of assignment[t]
                        long& asg = assignment[t];
                        if(asg != NULL_TOPIC){
     			     
			                --doc_topic_count[asg];
       			            --word_topic_count[asg];
        		            --global_topic_count[asg];
     			        }

		                // compute probability of multinomial
		                for(size_t t = 0; t < NTOPICS; ++t){
		                    const double n_dt = std::max(long(doc_topic_count[t]), long(0));
		                    const double n_wt = std::max(long(word_topic_count[t]), long(0));
		                    const double n_t  = std::max(long(global_topic_count[t]), long(0));
		                    prob[t] = (ALPHA + n_dt) * (BETA + n_wt) / (BETA * NWORDS + n_t);
		  		        }

                        // get new asg using random multinomial
		                asg = multinomial(prob);
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
		            for(size_t t = 0; t < NTOPICS; t++){
                        ms_send.factor[t] = 0;
                    }
		            for(size_t t = 0; t < assignment.size();t++){
                        ms_send.factor[assignment[t]]++;
                    }

                    sendMessageTo(vid_to, ms_send);

                }
                   
            }

        }else{// superstep != 0

            // get old vertex value
            val = getValue();

            // superstep odd: doc vertex work
		    if(getSuperstep() % 2 == 1){
       
                // doc recv assign, send factor to word
                if(val.flag==IS_DOC){

                    //receive new assignment from word vertex
                    for (;!pmsgs->done(); pmsgs->next()){	
                        assert(!pmsgs->done());
                        // add new assignment to old count factor
                        for(size_t t = 0 ; t < NTOPICS; t++){
			     			val.factor[t] += pmsgs->getValue().factor[t];	
                        }
                    }  

                    // send doc vertex factor to word
                    message_type ms_send;
                    ms_send.vid = getVertexId();
                    for(size_t t = 0; t < NTOPICS;t++){
                        ms_send.factor[t] = val.factor[t];
                    }
                    // printf("%ld\n",  ms_send.factor[1]);
                    // too many out messages, can't use sendall
                    // consider add worker or compress message
                    sendMessageToAllNeighbors(ms_send);
              
                }

                // compute aggregator
	            accumulateAggr(0, &val);

		    }

            // superstep even : word vertex work
		    if(getSuperstep() % 2 == 0){

                // check aggregator result and judge if voletohalt
				aggr_type aggr = *(aggr_type *)getAggrGlobal(0);
                long global_topic_count[NTOPICS];
                for(size_t t; t < NTOPICS; t++){
                    global_topic_count[t] = aggr.count[t];
                }
                double likelihood = aggr.likelihood;
				if(int64_t(getVertexId()) == int64_t(0)){
				    for(size_t t = 0; t < NTOPICS; t++){
				        printf("global_topic_count[%zu] = %ld\n", t, global_topic_count[t]);
				    }
				    printf("likelihood = %lf\n", likelihood);
				}

				// volt to halt
				if(getSuperstep() > 3){// if likelihood < ESP
				    voteToHalt(); return;   
				}


                // word recv message from doc,
                // update assignment of edge data
                // and send new assignment to doc
                if(val.flag == IS_WORD){

                    // get global topic count from aggregator
                    aggr_type aggr = *(aggr_type *)getAggrGlobal(0);
                    long word_topic_count[NTOPICS];
                    for(size_t t; t < NTOPICS; t++){
                        word_topic_count[t] = val.factor[t];
                    }

                    // each msg from one doc
                    // update the edge which connect current doc and this word
                    for (;!pmsgs->done();pmsgs->next()){

                         message_type recv = pmsgs->getValue();
                         unsigned long long vid_from = recv.vid;
                         long doc_topic_count[NTOPICS];
                         for(size_t t; t < NTOPICS; t++){
				             word_topic_count[t] = recv.factor[t];
                         }

                         OutEdgeIterator out_edge_it = getOutEdgeIterator();
                         //iterate outedges to find the edge connect to current doc
                         for (; !out_edge_it.done(); out_edge_it.next()){

                              unsigned long long vid_to = out_edge_it.target();
                              // if the edge is the one connect to current doc
                              // update the assignment on this edge    
                              // send message to current doc
                              if(vid_from == vid_to){ // break for

                                // point to edge data
                                Edge* edge = (Edge *)(out_edge_it.current());
                                edge_data* p = (edge_data *)(edge->weight);
                                vector_type assignment = p->assignment;

                                // probability of multinomial
                                std::vector<double> prob(NTOPICS);

						        for(size_t t = 0; t < assignment.size(); t++){

						            // asg is a referenc of assignment[t]
						            long& asg = assignment[t];
                                    if(asg != NULL_TOPIC){
			                            --doc_topic_count[asg];
       			                        --word_topic_count[asg];
        		                        --global_topic_count[asg];
                			        }

								    // compute probability of multinomial
								    for(size_t t = 0; t < NTOPICS; ++t){
								        const double n_dt = std::max(long(doc_topic_count[t]), long(0));
								        const double n_wt = std::max(long(word_topic_count[t]), long(0));
								        const double n_t  = std::max(long(global_topic_count[t]), long(0));
								        prob[t] = (ALPHA + n_dt) * (BETA + n_wt) / (BETA * NWORDS + n_t);
					  		        }

						            // get new asg using random multinomial
								    asg = multinomial(prob);
								    ++doc_topic_count[asg];
					  			    ++word_topic_count[asg];
					  			    ++global_topic_count[asg];

								}

						        // pass new assignment to current outedge data
						        for(size_t t = 0; t < assignment.size(); t++)
						            (p->assignment)[t] = assignment[t];

						        // send new assignment to doc vertex
							    message_type ms_send;
								ms_send.vid = getVertexId();
								for(size_t t = 0; t < NTOPICS; t++){
						            ms_send.factor[t] = 0;
						        }
								for(size_t t = 0; t < assignment.size();t++){
						            ms_send.factor[assignment[t]]++;
						        }

						        sendMessageTo(vid_to, ms_send);    
                                break; // end of outedge iterator
                              }
                         }
                    }
                }
		    }
        }

        // update vertex data
	    * mutableValue() = val;

    }
    int64_t get_outdegree(){
        int64_t rt = getOutEdgeIterator().size();
        return rt;
    }

    void init_edge_topic(){
        OutEdgeIterator outEdges = getOutEdgeIterator();
   
        for ( ; ! outEdges.done(); outEdges.next() ) {

            Edge* edge = (Edge *)(outEdges.current());
            edge_data* p = (edge_data *)(edge->weight);
            p->assignment.assign(p->ntoken, NULL_TOPIC);
        }
        return ;
    }

    void scatter_edge_topic(){
        OutEdgeIterator outEdges = getOutEdgeIterator();
        for ( ; ! outEdges.done(); outEdges.next() ) {
            Edge* edge = (Edge *)(outEdges.current());
            edge_data* p = (edge_data *)(edge->weight);
            //printf("ntoken = %zu\n", p->ntoken);
            for(size_t t = 0; t < (p->ntoken); t++){
                (p->assignment)[t] = t;
            }
        }
        return ;
    }

    void gather_edge_topic(){
        OutEdgeIterator outEdges = getOutEdgeIterator();
        for ( ; ! outEdges.done(); outEdges.next() ) {
            edge_data p = outEdges.getValue();
            //printf("ntoken = %zu\n", p.ntoken);
            for(size_t t = 0; t < (p.ntoken); t++){
                //printf("asg[%zu] = %ld\n", t,(p.assignment)[t]);
            }
        }
        return ;
    }
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
		for(double cumsum(prb[ind]/sum); 
			rnd > cumsum && (ind+1) < prb.size(); 
		cumsum += (prb[++ind]/sum));
		return ind;
    }

};

/** VERTEX_CLASS_NAME(Graph): set the running configuration here */
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

        setNumHosts(9);
        setHost(0, "localhost", 1411);
        setHost(1, "localhost", 1421);
        setHost(2, "localhost", 1431);
        setHost(3, "localhost", 1441);
        setHost(4, "localhost", 1451);
        setHost(5, "localhost", 1461);
        setHost(6, "localhost", 1471);
        setHost(7, "localhost", 1481);
        setHost(8, "localhost", 1491);

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

/* STOP: do not change the code below. */
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


