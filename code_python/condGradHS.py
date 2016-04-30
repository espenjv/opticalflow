
"""
Conditional gradient method for solving constrained quadratic optimization problem
with the poisson equation relating the state and the control.
"""



from dolfin import *

#Mesh and function spaces
N=50;
mesh =UnitSquareMesh(N,N)
V=FunctionSpace(mesh,"Lagrange",1)

#Y_omega,lambda, beta, f, y_analytic

lam=0.0351563

y_omega=Expression('2*(x[0]*x[0] +x[1]*x[1] -(x[0]+x[1])) + sin(pi*x[0])*sin(pi*x[1])')
y_omega=project(y_omega,V)

y_analytic=Expression('sin(pi*x[0])*sin(pi*x[1])')
y_bar=project(y_analytic,V)

#U_ad
u_a=Constant(0.0)
u_b=Constant(1.0)


# g is the characteristic function for the circle w. radius 1/4
class charfunc(Expression):
    def eval(selv,value,x):
        if (((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5)) <0.0625):
            value[0] = 1
        else:
            value[0] = 0



u_bar=charfunc()
u_bar=project(u_bar,V)

beta=Function(V)
beta=project(1-2*u_bar,V)

f=Function(V)
f=project(2*pi*pi*y_bar -beta*u_bar,V)


def descent(beta,p,u_a,u_b,u_n,lam):

    v_n=Function(V)
    P=p.vector().array()
    B=beta.vector().array()
    U=u_n.vector().array()
    v_na=v_n.vector().array()

    print P
    print B
    print U
    print v_na


    for i in range(len(u_n.vector().array())):
        if B[i]*P[i] + lam*U[i] > 0:
            v_na[i]=u_a

        elif B[i]*P[i] + lam*U[i] < 0:
            v_na[i]=u_b

        elif B[i]*P[i] + lam*U[i] == 0:
            v_na[i]=0.5*(u_a+u_b)

    print v_na
    v_n.vector()[:]=v_na
    print v_n.vector().array()
    v_n=project(v_n,V)
    print V.mesh().coordinates()
    print v_n.vector().array()
    return v_n


def s_min(u_n,v_n,y_omega,y_n,lam,f,beta,bc):

    w_n=Function(V)
    phi=TestFunction(V)
    w=TrialFunction(V)
    a_w=inner(nabla_grad(w),nabla_grad(phi))*dx
    L_w=(f +beta*v_n)*phi*dx
    solve(a_w==L_w,w_n,bc)

    g1=assemble(inner(y_n-y_omega,w_n-y_n)*dx)+ lam*assemble(inner(u_n,v_n-u_n)*dx)
    g2=0.5*assemble(inner(w_n-y_n,w_n-y_n)*dx) +0.5*lam*assemble(inner(v_n-u_n,v_n-u_n)*dx)

    s=-g1/(2*g2)
    s=min(1,max(s,0))
    return s



#Boundary conditions

def boundary(x,on_boundary):
    return on_boundary

bc_y=DirichletBC(V,Constant(0.0),boundary)
bc_p=DirichletBC(V,Constant(0.0),boundary)



#Initiating

#weak formulation

u_n=project(0.5*(u_a+u_b),V)

phi=TestFunction(V)
y=TrialFunction(V)

a_y=inner(nabla_grad(y),nabla_grad(phi))*dx
L_y=beta*u_n*phi*dx + f*phi*dx


y_n=Function(V)
solve(a_y==L_y,y_n,bc_y)

p=TrialFunction(V)

a_p=inner(nabla_grad(p),nabla_grad(phi))*dx
L_p=(y_n-y_omega)*phi*dx

p_n=Function(V)
solve(a_p==L_p,p_n,bc_p)

err=1
k=1
k_max=20

while err>DOLFIN_EPS and k<k_max:



    v_n=descent(beta,p_n,u_a,u_b,u_n,lam)
    print v_n.vector().array()
    s=s_min(u_n,v_n,y_omega,y_n,lam,f,beta,bc_y)
    print sqrt(s)
    u_n=project(u_n+s*(v_n-u_n),V)

    print u_n.vector().array()
    print v_n.vector().array()


    phi=TestFunction(V)
    y=TrialFunction(V)

    a_y=inner(nabla_grad(y),nabla_grad(phi))*dx
    L_y=beta*u_n*phi*dx + f*phi*dx

    y_n=Function(V)

    solve(a_y==L_y,y_n,bc_y)

    p=TrialFunction(V)

    a_p=inner(nabla_grad(p),nabla_grad(phi))*dx
    L_p=(y_n-y_omega)*phi*dx

    p_n=Function(V)

    solve(a_p==L_p,p_n,bc_p)
    uerr=project(u_n-u_bar,V)
    err=norm(uerr,'L2')
    k+=1
    print "L2-Error = %s" %err

    k = k_max

plot(u_n)
plot(u_bar)

interactive()
