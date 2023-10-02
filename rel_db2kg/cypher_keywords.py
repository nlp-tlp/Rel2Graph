from typing import Match




compound_sql_ops = [
        'intersect',
        'union',
        'except'
]

cypher_clauses = [
    'match',
    'merge',
    'optional', 
    'return',
    'union',
    'unwind',
    'with',
    'call',
    'create',
    'delete',
    'detach',
    'foreach',
    'load',
    'set',
    'start',
]

'''
Note: where operation is a subclause of match and with. 
when used as "with ... as new_variable, where new_variable ..", 
it is similar to the "having"

'''

cypher_subclauses = [
    'limit',
    'order',
    'skip',
    'where'  ]

cypher_modifiers = [
    'asc',
    'ascending',
    'assert',
    'by',
    'csv',
    'desc',
    'descending',
    'on'
]

cypher_expressions = [
    'all',
    'case',
    'else',
    'end',
    'then',
    'when'
]

cypher_operators = [
    'and',
    'as',
    'contains',
    'distinct',
    'ends',
    'in',
    'is',
    'not',
    'or',
    'starts',
    'xor'

]

cypher_reading_hints = [
    'using index',
    'using join',
    'using scan'
]

cypher_literals = [
    'false',
    'null',
    'true'
]

cypher_schema = [
    'constraint',
    'create',
    'drop',
    'exists',
    'index',
    'node',
    'key',
    'unique'
]

cypher_agg = [
    'avg',
    'max',
    'min',
    'sum',
    'count',
    'collect',
    'percentileCont',
    'percentileDisc',
    'stDev',
    'stDevP']
