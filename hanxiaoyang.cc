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

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>
//#include <boost/foreach.hpp>
//#include <boost/numeric/ublas/vector.hpp>
#include <assert.h>
#include <random>

#include "GraphLite.h"


#define DEBUG // run on vm


/** change VERTEX_CLASS_NAME(name) definition to use a different class name */
#define VERTEX_CLASS_NAME(name) cgs_lda##name

#define EPS 1e-6
/**
 * \brief the total number of topics to uses
 */

#ifdef DEBUG // run on vm
#define ZERO_TOPIC (long(0))
#endif

typedef unsigned long long vid_type;

#define NTOPICS 50 
size_t NWORDS = 0;

//typedef long count_type;
// factor_type is used to store the counts of tokens(word,doc pair) in each topic for words/docs.
typedef std::vector< long > factor_type;

//typedef uint16_t topic_id_type;
// We require a null topic to represent the topic assignment for tokens that have not yet been assigned.
#define NULL_TOPIC (uint16_t(-1))

/**
* The assignment type is used on each edge to store the
* assignments of each token.  There can be several occurrences of the
* same word in a given document and so a vector is used to store the
* assignments of each occurrence.
*/
typedef std::vector< long > assignment_type;

#define IS_WORD -1
#define IS_DOC 1
#define IS_NULL 0

double ALPHA = 1;
double BETA = 0.1;

typedef struct aggregator_struct
{
    double likelihood;
    long count[NTOPICS];
}aggr_type;

typedef struct
{
    //int a;
    long b[NTOPICS];
    long long c;//Vertex ID
}message_type;
/**
* The vertex data represents each word and doc in the corpus and contains 
* the counts of tokens(word,doc pair) in each topic. 
* change vertex_data from struct to typedef struct for sume bug of graphlite.
*/
typedef struct vertexData{
    // The count of tokens in each topic.
    factor_type factor;
    int flag;
    int64_t outdegree;
}vertex_data;

/**
* The edge data represents individual tokens (word,doc) pairs and their assignment to topics.
*/
typedef struct edgeData{
    size_t ntoken;
    // The assignment of all tokens
    assignment_type assignment;
}edge_data;

unsigned long long total_doc;
unsigned long long total_word;


/** VERTEX_CLASS_NAME(InputFormatter) can be kept as is */
class VERTEX_CLASS_NAME(InputFormatter): public InputFormatter {
public:
    int64_t getVertexNum() {
        unsigned long long n;
        sscanf(m_ptotal_vertex_line, "%lld %lld %lld", &n, &total_doc, &total_word);
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
/** VERTEX_CLASS_NAME(OutputFormatter): t
his is where the output is generated */
class VERTEX_CLASS_NAME(OutputFormatter): public OutputFormatter {
public:
    void writeResult() {
        int64_t vid;
        vertex_data value;
        char s[1024];

        for (ResultIterator r_iter; ! r_iter.done(); r_iter.next() ) {
            r_iter.getIdValue(vid, &value);
            int n = sprintf(s, "%lld:%d:%lld:%ld\n", vid, value.flag, value.outdegree, value.factor[0]);
            writeNextResLine(s, n);
        }
    }
};

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
        aggr_type aggr = *(aggr_type *) p;
        m_global.count[0] = aggr.count[0];
        for(int t = 0; t < NTOPICS; t++) m_global.count[t] = aggr.count[t];
    }
    void* getLocal() {
        return &m_local;
    }
    // type of p is <type name>
    void merge(const void* p) {
        aggr_type aggr = *(aggr_type *) p;
       for(int t = 0; t < NTOPICS; t++) m_global.count[t] += aggr.count[t];
       
    }
    // type of p is the type of value in AccumulateAggr(0, &value)
    void accumulate(const void* p) {
        vertex_data vdt = *(vertex_data*) p;
        for(int t = 0; t < NTOPICS; t++) m_local.count[t] += vdt.factor[t];
    }

};


/** VERTEX_CLASS_NAME(): the main vertex program with compute() */
class VERTEX_CLASS_NAME(): public Vertex <vertex_data, edge_data, message_type> {
public:
    void compute(MessageIterator* pmsgs) 
    {
        vertex_data val;
	assignment_type ass_send, ass_recv;
	message_type ms_send, ms_recv, mesa;
        //printf("%lld\n", getVertexId());
	if(getSuperstep() == 0)
	{
	    val.factor.assign(NTOPICS,0);
	    if(getVertexId() < total_doc) 
		val.flag = 1;
            else 
		val.flag = -1;
	    init_edge_topic();
	    if(val.flag == -1) 
	    {
		std::vector<double> prob(NTOPICS);

		long global[NTOPICS], doc_topic_count[NTOPICS], word_topic_count[NTOPICS];
		for(int k = 0; k < NTOPICS; k++)
		    global[k] = 0;
		for(int k = 0; k < NTOPICS; k++)
		    word_topic_count[k] = 0;
		for(int k = 0; k < NTOPICS; k++)
		    doc_topic_count[k] = 0;
		assignment_type assignment;

		//assignment.assign(NTOPICS,NULL_TOPIC);
		for (OutEdgeIterator out_edge_it = getOutEdgeIterator(); !out_edge_it.done(); out_edge_it.next())
		{
		    assignment.assign(out_edge_it.getValue().assignment.size(), 0);
		    for(int u = 0;u < assignment.size();u++)
			assignment[u] = out_edge_it.getValue().assignment[u];
		    long long vertex_id_to;
		    for(int u = 0;u < assignment.size();u++)
		    {
			uint16_t asg = assignment[u];
			const uint16_t old_asg = asg;
			if(asg != NULL_TOPIC) 
     			{
			    --doc_topic_count[asg];
       			    --word_topic_count[asg];
        		    --global[asg];
     			}
		    	for(size_t t = 0; t < NTOPICS; ++t)
     		    	{
        		    const double n_dt = std::max(long(doc_topic_count[t]), long(0));
        		    const double n_wt = std::max(long(word_topic_count[t]), long(0));
        		    const double n_t  = std::max(long(global[t]), long(0));
        		    prob[t] = (ALPHA + n_dt) * (BETA + n_wt) / (BETA * NWORDS + n_t);
      		    	}
	    	    	asg = multinomial(prob);
			assignment[u] = asg;
			++doc_topic_count[asg];
      			++word_topic_count[asg];
      			++global[asg];
		    }
		    ms_send.c = getVertexId();
		    for(int m = 0; m < NTOPICS;m++)
			ms_send.b[m] = 0;
		    for(int m = 0; m < assignment.size();m++)
			ms_send.b[assignment[m]]++;
		    sendMessageTo(vertex_id_to, ms_send);
		    //change edge data
		    Edge* edge = (Edge *)(out_edge_it.current());
            	    edge_data* p = (edge_data *)(edge->weight);
		    for(size_t t = 0; t < assignment.size(); t++)
               	 	(p->assignment)[t] = assignment[t];
		}
	    }
	    accumulateAggr(0, &val);
            * mutableValue() = val;
	}
        if(getSuperstep() % 2 && getSuperstep() != 0)
        {
	    val = getValue();

	    if(val.flag == -1) 
	    {
		std::vector<double> prob(NTOPICS);
		aggr_type global_topic_count = *(aggr_type *)getAggrGlobal(0);
             	//for(int t = 0; t < NTOPICS; t++) 
		//    printf("%ld\n",global_topic_count.count[t]);
		long global[NTOPICS], doc_topic_count[NTOPICS], word_topic_count[NTOPICS];
		for(int k = 0; k < NTOPICS; k++)
		    global[k] = global_topic_count.count[k];
		for(int k = 0; k < NTOPICS; k++)
		    word_topic_count[k] = getValue().factor[k];

		assignment_type assignment;	
			

		//gibbs sampling
		long long vertex_id_from, vertex_id_to;
		//size_t p;
		for (;!pmsgs->done();pmsgs->next()) 
		{
		    Edge* edge;
		    vertex_id_from = pmsgs->getValue().c;
		    for (OutEdgeIterator out_edge_it = getOutEdgeIterator(); !out_edge_it.done(); out_edge_it.next())
		    { 
		    	vertex_id_to = out_edge_it.target();
			if(vertex_id_from == vertex_id_to)
			{
			    edge = (Edge *)(out_edge_it.current());
			    assignment.assign(out_edge_it.getValue().assignment.size(), 0);
		    	    for(int u = 0;u < assignment.size();u++)
				assignment[u] = out_edge_it.getValue().assignment[u];
			    break;
			}
		    }
		    for(int k = 0; k < NTOPICS; k++)
		    	doc_topic_count[k] = pmsgs->getValue().b[k];
		    for(int u = 0;u < assignment.size();u++)
		    {
			uint16_t asg = assignment[u];
			const uint16_t old_asg = asg;
			if(asg != NULL_TOPIC) 
     			{ // construct the cavity
			    --doc_topic_count[asg];
       			    --word_topic_count[asg];
        		    --global[asg];
     			}
		    	for(size_t t = 0; t < NTOPICS; ++t)
     		    	{
        		    const double n_dt = std::max(long(doc_topic_count[t]), long(0));
        		    const double n_wt = std::max(long(word_topic_count[t]), long(0));
        		    const double n_t  = std::max(long(global[t]), long(0));
        		    prob[t] = (ALPHA + n_dt) * (BETA + n_wt) / (BETA * NWORDS + n_t);
      		    	}
	    	    	asg = multinomial(prob);
			assignment[u] = asg;
			++doc_topic_count[asg];
      			++word_topic_count[asg];
      			++global[asg];
		 	//p = out_edge_it.getValue().ntoken;
		    	//ass_send.assign(p, NULL_TOPIC); //superstep 0
		    	//for(int i = 0;i < ass_send.size();i++)
			//    ms_send.b[i] = ass_send[i];  
		    }
		    //send message
		    ms_send.c = getVertexId();
		    for(int m = 0; m < NTOPICS;m++)
			ms_send.b[m] = 0;
		    for(int m = 0; m < assignment.size();m++)
			ms_send.b[assignment[m]]++;
		    sendMessageTo(vertex_id_to, ms_send);
		    //change edge data
		    //Edge* edge = (Edge *)(out_edge_it.current());
            	    edge_data* p = (edge_data *)(edge->weight);
		    for(size_t t = 0; t < assignment.size(); t++)
               	 	(p->assignment)[t] = assignment[t];
        	}
	    }
	    accumulateAggr(0, &val);
            * mutableValue() = val;
        }
        if(getSuperstep() % 2 == 1)
	{    
	    if(getValue().flag == 1)
	    {
		val = getValue();
		for (;!pmsgs->done(); pmsgs->next())
	 	{		
		    for(int i = 0 ; i < NTOPICS; i++)			
			val.factor[i] += pmsgs->getValue().b[i];	
		}   
		memset(mesa.b, 0, NTOPICS); 
		for(int i = 0; i < NTOPICS; i++)
		    mesa.b[i] = val.factor[i];                
		mesa.c = getVertexId();    
		sendMessageToAllNeighbors(mesa);   
		* mutableValue() = val;
	    }
        }
	//determine stopping
	else 	
	{
             voteToHalt(); return;      
        }
        
        //val.outdegree = get_outdegree();
        //* mutableValue() = val;

    }
    // judge if a vertex is doc
    int is_doc(){
        return (getValue().flag==IS_DOC)? 1:0;
    }
    // judge if a vertex is word
    int is_word(){
        return (getValue().flag==IS_WORD)? 0:1;
    }
    int64_t get_outdegree(){
        int64_t rt = getOutEdgeIterator().size();
        return rt;
    }
    // count tokens from edges
    size_t count_tokens(){
        // count number of tokens on edges 
        size_t ntokens = 0;
        //int64_t vid = getVertexId();
        //OutEdgeIterator out_edge_it = getOutEdgeIterator();
        //for ( ; ! out_edge_it.done(); out_edge_it.next() ) {
            //ntokens += (out_edge_it.getValue()).assignment.size();
        //}
        //printf("vid=%lld, ntokens=%zu\n", vid, ntokens);
        return ntokens;
    }

    size_t multinomial(const std::vector<double>& prb) 
    {
	std::default_random_engine generator;
  	std::uniform_real_distribution<double> distribution(0.0,1.0);
	assert(prb.size()>0);
	if (prb.size() == 1) 
	    return 0;
	double sum(0);
	for(size_t i = 0; i < prb.size(); ++i) 
	{
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
    
    void init_edge_topic(){
        OutEdgeIterator outEdges = getOutEdgeIterator();
   
        for ( ; ! outEdges.done(); outEdges.next() ) {

            Edge* edge = (Edge *)(outEdges.current());
            edge_data* p = (edge_data *)(edge->weight);
            p->assignment.assign(p->ntoken, NULL_TOPIC);
        }
        return ;
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

        setNumHosts(5);
        setHost(0, "localhost", 1411);
        setHost(1, "localhost", 1421);
        setHost(2, "localhost", 1431);
        setHost(3, "localhost", 1441);
        setHost(4, "localhost", 1451);

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






