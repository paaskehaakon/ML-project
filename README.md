# ML-project

# questions for ta
- Better way to generate dummief for categorical data rater than pd.get_dummies()?
- Tips to my model. how can i make it better?
- what framework could i use to make it better? 
- what to use stores extra for?


# READING MATERIAL
https://www.kaggle.com/code/lasmith/house-price-regression-with-lightgbm
https://www.kaggle.com/code/arsenal/geomap-for-average-revenue/script
https://www.kaggle.com/code/jquesadar/restaurant-revenue-1st-place-solution
https://www.kaggle.com/code/artgor/eda-feature-engineering-and-model-interpretation#Data-exploration
https://www.kaggle.com/competitions/restaurant-revenue-prediction/code

Tips from TA: https://www.kaggle.com/code/pmarcelino/comprehensive-data-exploration-with-python


# Plan moving forward
 - implement a simple ml
 - google our way thorugh similar project to get inspiration on how to tacket the course
 - iterate upon our design

# deadline 19.10
haakon: Have set up a lightgbm model that can be used to validate feature engineering on. 
julia: Complete guide. (https://www.kaggle.com/code/ayushikaushik/eda-regression-analysis#Preprocessing-the-data)

# deadline 21.10
haakon: geo features
julia: non geo features

# deadline 26.10
haakon: nbr stores in same lv2 within radius
julia: read article, other geo features


# notes 26.10
vektorisera med numpy
balltree göra vektoriserat
balltree för alla buss stops, balltree för varje viktighetsnivå

category encoders - cat boost. sklearn interface. Pass på overfitting.

Mappa alla y-värden till ett space, predikera där, mappa tillbaka till vanliga y spacet innan submission till kaggle

# suggestions for data manipulation
- bussstop
    - find number of bussstops close to shop
    - represent importance level of bussstop and number of bussstops to extract some numbers
- age distributino
    - somehow use the dempgraphic overview to get a number on rating
- houshold num persons
    - maybe leave as is. integrate with what shops lie within what grunnkrets
- income houshold
    - again, use with what shops lie wheinin what grunnkrets to get good numbers for data
- grunnkrets
    - dont know
- place hierarcy
    - more data on the shops. Should find a way to integrate
- stores extra
    - leave as is. dont know what to do with it
- stores test
    - dont know what to do with it. 

