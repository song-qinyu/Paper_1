function Bik_u = BaseFunction(i,k,u,NodeVector)
%基函数    Bik
 
    if k==0
        if u>= NodeVector(i+1) && u<NodeVector(i+2)
            Bik_u = 1;
        else
            Bik_u = 0;
        end
    else
        Length1=NodeVector(i+k+1)-NodeVector(i+1);
        Length2=NodeVector(i+k+2)-NodeVector(i+2);
        if Length1 == 0
            Length1=1;
        end
        if Length2 == 0
            Length2 = 1;
        end
        % 递归 Bik_u=（u-u_i）/(u_i+k-u_i)*B_i_k-1_u
        %             +(u_i+k+1-u)/(u_i+k+1-u_i+1)* B_i+1_k-1_u
        Bik_u=(u-NodeVector(i+1))/Length1 * BaseFunction(i,k-1,u,NodeVector)...
            +(NodeVector(i+k+2)-u)/Length2 * BaseFunction(i+1,k-1,u,NodeVector);
    end
 