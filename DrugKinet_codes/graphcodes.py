import cvxpy as cvx
import numpy as np
from scipy.linalg import sqrtm
def graph_reg(Y, mask, Uknn, Vknn): #varargin: nLatentFactors, 
    k = 20
    maxIterations = 10
    n,m = np.shape(Y)
    V = np.random.normal(0,1,(k,m))
    prev_error = 0
    c = 1 #regularixer coefficient
    #Changed just now
    #Uknn = Usim;
    #Vknn = Vsim;
    
    
    Du = np.diag(sum(Uknn,1));  
    LapU = Du - Uknn;
    Dv = np.diag(sum(Vknn,1)); 
    LapV = Dv - Vknn;
    
    lamb = 1;
    lambU = 1;
    lambV = 1;
    Lu = lambU* LapU + lamb*np.eye(n);
    Lv = lambV* LapV + lamb*np.eye(m);
    sqrtLu = np.real(sqrtm(Lu));
    sqrtLv = np.real(sqrtm(Lv));
    

    for i in range(maxIterations):
        if(i%2 == 0):
            #Put V constant
            U = cvx.Variable(n,k)
            reg = sqrtLu*U;
            regC = np.matmul(V,sqrtLv);
        else:
            #Put U constant
            V = cvx.Variable(k,m)
            reg = V*sqrtLv;
            regC = np.matmul(sqrtLu,U);
            
        error = Y[mask==1] - (U*V)[mask==1]
        #error = Y - (U*V)
        
        obj = cvx.Minimize(cvx.norm(error, 'fro')**2 + c*cvx.norm(reg, 'fro') + cvx.norm(regC, 'fro')) 
        prob = cvx.Problem(obj)
        prob.solve(solver=cvx.SCS)
        print(prob.value)
        
        if(i%2 == 0):
            U = U.value
        else:
            V = V.value
            
        #Convergence condition
        if(abs(prev_error - prob.value) < 0.1 ):
            break
        prev_error = prob.value
    Yhat = np.matmul(U, V)
    return Yhat,U,V


#%%
def biasedgraph_reg(Y, mask, Uknn, Vknn): #varargin: nLatentFactors, 
    k = 25
    maxIterations = 15
    n,m = np.shape(Y)
    V = np.random.normal(0,1,(k,m))
    bv = np.random.normal(0,1,(1,m))
    r1 = np.ones([1,m]); r2 = np.ones([n,1])
    prev_error = 0
    c = 1 #regularixer coefficient
    #Changed just now
    #Uknn = Usim;
    #Vknn = Vsim;
    
    
    Du = np.diag(sum(Uknn,1));  
    LapU = Du - Uknn;
    Dv = np.diag(sum(Vknn,1)); 
    LapV = Dv - Vknn;
    
    lamb = 1;
    lambU = 0;
    lambV = 0;
    Lu = lambU* LapU + lamb*np.eye(n);
    Lv = lambV* LapV + lamb*np.eye(m);
    sqrtLu = np.real(sqrtm(Lu));
    sqrtLv = np.real(sqrtm(Lv));
    

    for i in range(maxIterations):
        if(i%2 == 0):
            #Put V constant
            U = cvx.Variable(n,k)
            bu = cvx.Variable(n,1)
            reg = sqrtLu*U;
            regC = np.matmul(V,sqrtLv);
        else:
            #Put U constant
            V = cvx.Variable(k,m)
            bv = cvx.Variable(1,m)
            reg = V*sqrtLv;
            regC = np.matmul(sqrtLu,U);
            
        error = Y[mask==1] - (U*V)[mask==1] - (bu*r1)[mask==1] - (r2*bv)[mask==1]
        #error = Y - (U*V)
        
        #if(i%2 == 0):
        obj = cvx.Minimize(cvx.norm(error, 'fro')**2 + cvx.norm(reg, 'fro')**2 + cvx.norm(regC, 'fro')**2 + 1.0/n * cvx.norm(bu)**2 + 1.0/m *cvx.norm(bv)**2) 
        #else:
            #obj = cvx.Minimize(cvx.norm(error, 'fro') + cvx.norm(reg, 'fro') + cvx.norm(regC, 'fro') + cvx.norm(bu) + cvx.norm(bv))
        prob = cvx.Problem(obj)
        prob.solve(solver=cvx.SCS)
        print(prob.value)
        
        if(i%2 == 0):
            U = U.value
        else:
            V = V.value
            
        #Convergence condition
        if(abs(prev_error - prob.value) < 0.1 ):
            break
        prev_error = prob.value
    bu = bu.value
    bv = bv.value
    Yhat = np.matmul(U, V) + np.matmul(bu,r1) + np.matmul(r2,bv)
    embed = np.matmul(U, V)
    bias = np.matmul(bu,r1) + np.matmul(r2,bv)
    return Yhat, embed, bias, U, V, bu, bv


#%%
def biasedgraph_reg2(Y, mask, Uknn, Vknn): #varargin: nLatentFactors, 
    k = 20
    maxIterations = 10
    n,m = np.shape(Y)
    V = np.random.normal(0,1,(k,m))
    bv = np.random.normal(0,1,(1,m))
    r1 = np.ones([1,m]); r2 = np.ones([n,1])
    prev_error = 0
    c = 1 #regularixer coefficient
    #Changed just now
    #Uknn = Usim;
    #Vknn = Vsim;
    
    
    Du = np.diag(sum(Uknn,1));  
    LapU = Du - Uknn;
    Dv = np.diag(sum(Vknn,1)); 
    LapV = Dv - Vknn;
    
    lamb = 1;
    lambU = 1;
    lambV = 1;
    Lu = lambU* LapU + lamb*np.eye(n);
    Lv = lambV* LapV + lamb*np.eye(m);
    sqrtLu = np.real(sqrtm(Lu));
    sqrtLv = np.real(sqrtm(Lv));
    

    for i in range(maxIterations):
        if(i%2 == 0):
            #Put V constant
            U = cvx.Variable(n,k)
            bu = cvx.Variable(n,1)
            reg = sqrtLu*U;
            regC = np.matmul(V,sqrtLv);
        else:
            #Put U constant
            V = cvx.Variable(k,m)
            bv = cvx.Variable(1,m)
            reg = V*sqrtLv;
            regC = np.matmul(sqrtLu,U);
            
        error = Y[mask==1] - (U*V)[mask==1] - (bu*r1)[mask==1] - (r2*bv)[mask==1]
        #error = Y - (U*V)
        
        if(i%2 == 0):
            obj = cvx.Minimize(cvx.norm(error, 'fro') + c*cvx.norm(reg, 'fro') + cvx.norm(regC, 'fro')) 
        else:
            obj = cvx.Minimize(cvx.norm(error, 'fro') + c*cvx.norm(reg, 'fro') + cvx.norm(regC, 'fro'))
        prob = cvx.Problem(obj)
        prob.solve(solver=cvx.SCS)
        print(prob.value)
        
        if(i%2 == 0):
            U = U.value
        else:
            V = V.value
            
        #Convergence condition
        if(abs(prev_error - prob.value) < 0.1 ):
            break
        prev_error = prob.value
    bu = bu.value
    bv = bv.value
    Yhat = np.matmul(U, V) + np.matmul(bu,r1) + np.matmul(r2,bv)
    return Yhat, U, V, bu, bv    