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


# notes 26.10 - To do:
- Use balltree/numpy instead of nested for loops in geo features - Haakon jobbar med detta
- Category encoders instead of dummy variables - cat boost. sklearn interface. Pass på overfitting.
- Map all y to one space, predict in that space, map back to original space before submission to Kaggle
- Revenue of stores in same lv2 in a radius of x - Julia jobbar med detta
- Make model - gör på torsdag 27.10

# notes 27.10 - To do:
- Kolla med Ruslan att alla groupby etc är rätt
- Ta log på rätt ställe på revenue värdena, obs att inte ta log både på revenue värdena i början och i uträkningen av RMSLE - gjort
- Fixa correlation plot - Haakon har gjort
- Fixa categorical features av hierarchy levelsen - gjort

# notes from Ruslan
- Gör catboost eller lightgbm med enkla features, försök inte göra categorical encoding själva utan låt modellen sköta det först. Om log transformen fram och tillbaka är bra borde vi slå 5 virtual teams.
- mean revenue per chain name är en dålig feature eftersom det är många chains med bara en store_id i, då blir det data leakage. Gör sådana features för grejer där det finns många store_ids i varje kategori, då aggregeras det. Om det är för få store_ids i varje grupp kopierar den bara = target leakage.

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

# questions ta
- hvordan bruke kdtree/finne avstand fra punkt til punkt. 

# TODO HAAKON
- fikse sånn at csv filene lagres i en fin mappe etter hvert steg. så sjekker funksjonen om csv filen finnes. hvis den gjør det så hopper den over fuksjon steget. 
- meadian income for households within 2 km
- closest competitor
- number of stores within 10km. 
- number oh households within 2 km
- percentage of households with income over x krones within y distance from shop
- average age within x distance from shop
- family size

