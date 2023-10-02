import json
from pygments.lexers import get_lexer_by_name
import os
from pathlib import Path
here = Path(__file__).parent

class CyqueryStatmentParser:
	def __init__(self, 
				queries, 
				queries_type,
				lexer):
		self.queries = queries
		self.queries_type = queries_type
		self.lexer = lexer

	def set_key(self, 
				dictionary, 
				key, 
				value):
		if key not in dictionary:
			dictionary[key] = value
		elif type(dictionary[key]) == list and value not in dictionary[key]:
			dictionary[key].append(value)
		elif value not in dictionary[key]:
			dictionary[key] = [dictionary[key], value]

		return dictionary
			
	def save2file(self, ls_name, ls, mode):
		folder ='{}/{}/{}/'.format(here, "output", "cypher")
		if not os.path.exists(os.path.dirname(folder)):
			try:
				os.makedirs(os.path.dirname(folder))
			except OSError as exc:
				if exc.errno != errno.EEXITST:
					raise		
		if isinstance(ls, list):
			with open(folder + ls_name, mode) as out:
				print(ls, file = out)
		else:
			with open(folder + ls_name, mode) as f:
				json.dump(ls, f, indent = 6)
				f.close()


	def get_tokenization(self):
		if self.queries_type=='file':
			with open(self.queries, 'r') as f:
				queries = f.read()
			queries = list(queries.split(';'))
		elif self.queries_type=='statement':
			queries=[self.queries]
		# print("queries:", queries, type(queries))
		token_types = []
		tokenized_statment = {}
		for query in queries:
			get_tokens = list(self.lexer.get_tokens(query))			
			self.save2file('get_tokens.txt', get_tokens, 'a')
			for every in get_tokens:
				token_type = str(every[0])
				token_split = token_type.split(".")
				key = '_'.join(token_split[:])	
				tokenized_statment = self.set_key(tokenized_statment, key, every[1])
				if every[0] not in token_types:
					token_types.append(every[0])


		self.save2file('get_lexer', token_types, 'w')
		self.save2file('tokenized_statment.json', tokenized_statment, 'w')
		
		return tokenized_statment, token_types

def main():

	lexer = get_lexer_by_name("py2neo.cypher")
	
	queries_file = 'input/queries'
	Cyparser = CyqueryStatmentParser(queries_file, lexer)
	tokenized_statment, token_types = Cyparser.get_tokenization()
	print(f'tokenized_statment: {tokenized_statment}')
	print(f'token_types: {token_types}')


if __name__ == "__main__":
	main()
