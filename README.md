# ZapposChallenge
This Repository includes file submission for Zappos Challenge.  

## DESCRIPTIONS OF DATA FILE    
  
__u.data__  
  
This data set consists of:  
	1. 100,000 visits (1-5) from 943 users on 1682 products  
	2. Each user has visited at least 20 items  
        3. Simple demographic info for the users (age, gender, occupation, zip)  
	4. Users and items are numbered consecutively from 1. The data is randomly ordered. This is a tab separated list of user id | item id | visits | timestamp.  
	The time stamps are unix seconds since 1/1/1970 UTC  
Note: The original dataset has been created for some different purpose.But I am modifying it and making it for product recommandation.  

## DESCRIPTIONS OF OTHER FILES
  
1. recommendation.py : A recommendation engine created on python 2.7 using the above dataset. It requires the dataset as an input and generates the 5 recommended products for a user_2 and user_10. The output can be generated for other users as well simply. It also shows the root mean square error for the predicted results.  
2. Zappos Challenge.docx : A complete submission of all parts of the challenge. It includes algorithms,implemetations and measureements of the recommendations.
