import graphlab
# user -> object
# item -> classifier
# rating -> possibility of correct classification
sf = graphlab.SFrame({'user_id': ["0", "0", "0", "1", "1", "1"],
                       'item_id': ["cl1", "cl2", "cl3", 
                                   "cl1", "cl2", "cl3"],
                      'rating': [.5, .2, .6, 
                                 .3, .8, .4]})
user_info = graphlab.SFrame({'user_id': ['0', '1'],
                             'm1': [1, 2], 'm2': [2, 3]})
m_side_info = graphlab.factorization_recommender.create(sf, target='rating',
                                                        user_data=user_info)
new_user_info = graphlab.SFrame({'user_id' : ['2'],
                                 'm1': [1], 'm2': [2]})
recommendations = m_side_info.recommend(['2'],
                                        new_user_data = new_user_info)
print recommendations
