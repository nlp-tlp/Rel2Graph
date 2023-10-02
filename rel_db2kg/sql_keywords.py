from pyparsing import Keyword, MatchFirst
from debugs import debug
ND = None
AS = None
ASC = None
BETWEEN = None
CASE = None
COLLATE_NOCASE = None
CROSS_JOIN = None
DESC = None
END = None
ELSE = None
FROM = None
FULL_JOIN = None
FULL_OUTER_JOIN = None
GROUP_BY = None
HAVING = None
IN = None
INNER_JOIN = None
IS = None
IS_NOT = None
JOIN = None
LEFT_JOIN = None
LEFT_OUTER_JOIN = None
LIKE = None
LIMIT = None
NOT_BETWEEN = None
NOT_IN = None
NOT_LIKE = None
OFFSET = None
ON = None
OR = None
ORDER_BY = None
RESERVED = None
RIGHT_JOIN = None
RIGHT_OUTER_JOIN = None
SELECT = None
THEN = None
UNION = None
UNION_ALL = None
EXCEPT = None
INTERSECT = None
USING = None
WITH = None
WHEN = None
WHERE = None

locs = locals()
reserved = []
for name, v in list(locs.items()):
    if v is None:
        n = name.lower().replace("_", " ")
        value = locs[name] = Keyword(n, caseless=True).setName(n).setDebugActions(*debug)
        reserved.append(value)
RESERVED = MatchFirst(reserved)

sql_unordered_clauses = [
    "with",
    "select",
    "from",
    "where",
    "groupby",
    "having",
]

sql_ordered_clauses = [
    "orderby",
    "limit",
    "offset",
]


sql_join_keywords = [
    "join",
    "full join",
    "cross join",
    "inner join",
    "left join",
    "right join",
    "full outer join",
    "right outer join",
    "left outer join",
]




