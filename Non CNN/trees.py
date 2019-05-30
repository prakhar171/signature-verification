# Python program to insert element in binary tree  
class newNode():  
  
    def __init__(self, data):  
        self.key = data 
        self.left = None
        self.right = None
          
""" Inorder traversal of a binary tree"""
def inorder(temp): 
  
    if (not temp): 
        return temp
  
    inorder(temp.left)  
    print(temp.key,end , " ") 
    inorder(temp.right)  
  
  
"""function to insert element in binary tree """
def insert(temp,key): 
  
    q = []  
    q.append(temp)  
  
    # Do level order traversal until we find  
    # an empty place.  
    while (len(q)):  
        temp = q[0]  
        q.pop(0)  
  
        if (not temp.left): 
            temp.left = newNode(key)  
            break
        else: 
            q.append(temp.left)  
  
        if (not temp.right): 
            temp.right = newNode(key)  
            break
        else: 
            q.append(temp.right)  

def delete(temp, key):
    # Find the rightmost bottommost element.

    new_last = None
    if (not temp):
        return

    inorder(temp)
    # inorder(temp.right)
    new_last = temp.key
    # print("\n", new_last)

# Driver code  
if __name__ == '__main__': 
    root = newNode(13)  
    root.left = newNode(12)  
    root.left.left = newNode(4)
    root.left.right = newNode(19)  
    root.right = newNode(10)  
    root.right.left = newNode(16)  
    root.right.right = newNode(9)  
  
    print("Inorder traversal before Deletion:", end , " ") 
    inorder(root)  
  
    key = 10
    delete(root, key)  
  
    # print()  
    # print("Inorder traversal after insertion:", end = " ") 
    # inorder(root) 
    # print()
  
# This code is contributed by SHUBHAMSINGH10