
class GraphDataModel():
    '''
    Input: a property graph 𝐺 = (𝑉, 𝐸, st, 𝐿, 𝑇, ℒ, 𝒯, 𝑃𝑣, 𝑃𝑒), where
        - 𝑉: a set of vertices
        - 𝐸: a set of edges
        - st: E -> V x V assigns source and target vertices to edges
        - 𝐿: a set of vertex labels
        - 𝑇: a set of edge types
        - ℒ: 𝑉 → 2_power(𝐿) assign a set of labels to each vertex
        - 𝒯 : 𝐸 → 𝑇 assign a single type to each edge

        Let 𝐷 = ∪𝑖𝐷𝑖 be the union of atomic domains 𝐷𝑖.
        Let 𝜀 = NULL value.    
        - 𝑃𝑣: a set of vertex properties
            𝑝𝑖 ∈ 𝑃𝑣 is a partial function 𝑝𝑖 : 𝑉 → 𝐷𝑖 ∪ {𝜀} assigns a property value from a domain 𝐷𝑖 ∈ 𝐷 
            to a vertex 𝑣 ∈ 𝑉, if 𝑣 has property 𝑝𝑖, otherwise 𝑝𝑖(𝑣) returns 𝜀.
        - 𝑃𝑒: a set of edge properties

    Output: a graph relation
    '''        
    def __init__(self, vertex, edge, label, type, property_v, property_e):
        self.vertex = vertex
        self.edge = edge
        self.label = label
        self.type = type
        self.property_v = property_v
        self.property_e = property_e

    def st(self):
        '''
        graph patterns
        '''
        raise NotImplementedError
        
    @classmethod
    def assign_lable(cls):
        '''
        :return the labels of vertex v.
        '''
        raise NotImplementedError

    
    @classmethod
    def assign_edge(cls):
        '''
        :return the types of edge e.
        '''
        raise NotImplementedError




class GraphEngine(object):
    '''
    An abstract engine base class.
    '''

    def __init__(self, model: 'dataModel'):
        '''
        :parameter model: The model that this engine will solve quries for.
        '''
        self.model = model
    
    def prepare(self, query: LogicProgram):

        '''
        Verfify the given program to a format suited for querying in this engine.
        :parameter query: The query to be verified
        :return The verified query
        '''
        
        raise NotImplementedError('prepare is an abstract method')


    





