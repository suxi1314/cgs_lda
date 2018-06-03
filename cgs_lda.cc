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

#include "GraphLite.h"


#define DEBUG // run on vm


/** change VERTEX_CLASS_NAME(name) definition to use a different class name */
#define VERTEX_CLASS_NAME(name) cgs_lda##name

#define EPS 1e-6
/**
 * \brief the total number of topics to uses
 */

#ifdef DEBUG // run on vm
size_t NTOPICS = 2;
size_t NTOKEN = 2;
#define ZERO_TOPIC (topic_id_type(0))
#endif

#ifdef RELEASE // run on server
size_t NTOPICS = 50; 
#endif

typedef long count_type;
// factor_type is used to store the counts of tokens(word,doc pair) in each topic for words/docs.
typedef std::vector< count_type > factor_type;

typedef uint16_t topic_id_type;
// We require a null topic to represent the topic assignment for tokens that have not yet been assigned.
#define NULL_TOPIC (topic_id_type(-1))

/**
* The assignment type is used on each edge to store the
* assignments of each token.  There can be several occurrences of the
* same word in a given document and so a vector is used to store the
* assignments of each occurrence.
*/
typedef std::vector< topic_id_type > assignment_type;

/**
* The vertex data represents each word and doc in the corpus and contains 
* the counts of tokens(word,doc pair) in each topic. 
*/
struct vertex_data{
    // The count of tokens in each topic.
    factor_type factor;
    vertex_data() : factor(NTOPICS) { }

};

/**
* The edge data represents individual tokens (word,doc) pairs and their assignment to topics.
*/
struct edge_data{
    // The assignment of all tokens
    assignment_type assignment;
    edge_data(size_t ntoken = 0) : assignment(ntoken, NULL_TOPIC) { }
};


/** VERTEX_CLASS_NAME(InputFormatter) can be kept as is */
class VERTEX_CLASS_NAME(InputFormatter): public InputFormatter {
public:
    int64_t getVertexNum() {
        unsigned long long n;
        sscanf(m_ptotal_vertex_line, "%lld", &n);
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
        m_m_value_size = sizeof(double);
        return m_m_value_size;
    }
    void loadGraph() {
        unsigned long long last_vertex;
        unsigned long long from;
        unsigned long long to;

        edge_data weight = 0;
        size_t ntoken = 0;

        vertex_data value = vertex_data();
        int outdegree = 0;
        

        const char *line= getEdgeLine();

        // Note: modify this if an edge weight is to be read
        //       modify the 'weight' variable

        // read edge weight
        sscanf(line, "%lld %lld %zu", &from, &to, &ntoken);
#ifdef DEBUG
        weight = edge_data(NTOKEN);
#endif

#ifdef RELEASE
        weight = edge_data(ntoken);
#endif

        // change word vertex vid
        to = m_total_vertex + to;

        addEdge(from, to, &ntoken);

        last_vertex = from;
        ++outdegree;
        for (int64_t i = 1; i < m_total_edge; ++i) {
            line= getEdgeLine();

            // Note: modify this if an edge weight is to be read
            //       modify the 'weight' variable

            // read edge weight
            sscanf(line, "%lld %lld %zu", &from, &to, &ntoken);
#ifdef DEBUG
            weight = edge_data(NTOKEN);
#endif

#ifdef RELEASE
            weight = edge_data(ntoken);
#endif
            // change word vertex vid
            to = m_total_vertex + to;

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
/** VERTEX_CLASS_NAME(OutputFormatter): this is where the output is generated */
class VERTEX_CLASS_NAME(OutputFormatter): public OutputFormatter {
public:
    void writeResult() {
        int64_t vid;
        vertex_data value;
        char s[1024];

        for (ResultIterator r_iter; ! r_iter.done(); r_iter.next() ) {
            r_iter.getIdValue(vid, &value);
            int n = sprintf(s, "%lld: \n", (unsigned long long)vid);
            writeNextResLine(s, n);
        }
    }
};

/** VERTEX_CLASS_NAME(Aggregator): you can implement other types of aggregation */
// An aggregator that records a double value to m compute sum
class VERTEX_CLASS_NAME(Aggregator): public Aggregator<double> {
public:
    void init() {
        m_global = 0;
        m_local = 0;
    }
    void* getGlobal() {
        return &m_global;
    }
    void setGlobal(const void* p) {
        m_global = * (double *)p;
    }
    void* getLocal() {
        return &m_local;
    }
    void merge(const void* p) {
        m_global += * (double *)p;
    }
    void accumulate(const void* p) {
        m_local += * (double *)p;
    }
};

/** VERTEX_CLASS_NAME(): the main vertex program with compute() */
class VERTEX_CLASS_NAME(): public Vertex <vertex_data, edge_data, double> {
public:
    void compute(MessageIterator* pmsgs) {
        // output number of outedge 
        vertex_data val = vertex_data();
        if (getSuperstep() == 0) {
            size_t ntokens = collect();
        } else {
            voteToHalt(); return;
        }
        * mutableValue() = val;
    }
    // judge if a vertex is doc
    int is_doc(){
        return (getOutEdgeIterator().size())? 1:0;
    }
    // judge if a vertex is word
    int is_word(){
        return (getOutEdgeIterator().size())? 0:1;
    }
    // collect assignment from edges
    size_t collect(){
        // count number of tokens on out edges 
        size_t ntokens;
        int64_t vid = getVertexId();
        OutEdgeIterator outEdges = getOutEdgeIterator();
        for ( ; ! outEdges.done(); outEdges.next() ) {
            ntokens += (outEdges.getValue()).assignment.size();
        }
        printf("vid=%lld, ntokens=%zu\n", vid, ntokens);
        return ntokens;
    }

    void assignment(){
        int64_t vid = getVertexId();
        OutEdgeIterator outEdges = getOutEdgeIterator();
        for ( ; ! outEdges.done(); outEdges.next() ) {
            char* p = outEdges.current();
            topic_id_type topic = NULL_TOPIC;
#ifdef DEBUG
            topic = ZERO_TOPIC;
#endif
            assignment_type *assignment;
            * assignment = ((assignment_type)((( edge_data *)( (Edge *)p )->weight) -> assignment));
            size_t assignment_size = assignment->size();
            assignment->assign(assignment_size,ZERO_TOPIC);
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


