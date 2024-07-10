'''In an e-commerce company, the management wants to predict whether a customer will purchase a high-value product based on their age, time spent on the website, and whether they have added items to their cart. The goal is to optimize marketing strategies by targeting potential customers more effectively, thereby increasing sales and revenue.'''
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X=np.array([[25,30,0],[30,40,1],[20,35,0],[35,45,1]]) # ['age','time spent','yes or no']
y=np.array([0,1,0,1]) # extraction of 3rd argument

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LogisticRegression()

model.fit(X_train,y_train)

accuracy=model.score(X_test,y_test) # whether testing is right or wrong
print(f"model accuracy:{accuracy}")

user_age=float(input("Enter customer age: "))
user_time_spent=float(input("Enter time spent on website: "))
user_added_to_cart=int(input("Enter 1 if added to cart,else enter 0: "))
user_Data=np.array([[user_age, user_time_spent, user_added_to_cart]])

prediction=model.predict(user_Data)

if prediction[0]==1:
    print("The customer is likely to purchase")
else:
    print("The customer is unlikely to purchase")