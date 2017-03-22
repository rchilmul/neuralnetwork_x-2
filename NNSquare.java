//CS4811 ARTIFICIAL INTELLIGENCE SPRING 2017
//NEURAL NETWORK PROJECT (X^2)
//Author: RAVIKUMAR CHILMULA
//Date: 2/3/2017


/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nnsquare;

import java.awt.Color;
import java.awt.GridLayout;
import java.util.ArrayList;
import java.util.Random;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JTextField;

/**
 *
 * @author chravikumar
 */
public class NNSquare extends JFrame{

 static ArrayList<Integer> FinalResult= new ArrayList<Integer>();
 static ArrayList<Integer> Final= new ArrayList<Integer>();
 static ArrayList<Double> FinalInputX= new ArrayList<Double>();
 static ArrayList<Double> FinalInputY= new ArrayList<Double>();
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        //Initilizing the input
        Random randomGenerator= new Random();
        double [] []wH1 = new double[3][2];    //Weights for hidden layer 1 inputs
        double [] []wH2 = new double[3][2];     //Weights for hidden layer 2 inputs
        double [] []Output = new double[101][101];
        double []wO = new double[3];          //Weights for Output layer inputs
        //Training Data Initialization
        double[][] inputIN= 
        {{0,0,0,0.1,0.1,0.2,0.2,0.2,0.3,0.4,0.4,0.5,0.5,0.6,0.6,0.7,0.7,0.8,0.8,0.9,0.9,1},
         {0.1,0.5,0.9,0.01,0.5,0,0.1,1,0.09,0.1,0.17,0.2,0.3,0.3,0.36,0.4,0.5,0.2,0.8,0.8,0.9,0.9}};//Input Training Examples(X^2)
        int[] desired={0,0,0,1,0,1,0,0,1,1,0,1,0,1,1,1,0,1,0,1,0,1};              //Desired output
        
//Declaration of 4 layers(input,hidden1,hidden2,output) with IN and OUT of each perceptron
        double[] inputOUT = new double[3]; 
        double[] hidden1IN = new double[2], hidden2IN = new double[2];
        double [] hidden1OUT = new double[3], hidden2OUT = new double[3];
        double [] deltaH1 = new double[2], deltaH2 = new double[2]; 
        double outputIN, outputOUT;
        int bias=1;                    //Considering bias
        System.out.println("Feed Forward Neuron Network ");
        //Initialisation of Weights zero
        wH1[0][0] =0;
        wH1[0][1] =0;
        wH1[1][0] =0;
        wH1[1][1] =0;
        wH1[2][0] =0;
        wH2[2][1] =0;
        wH1[0][0] =0;
        wH2[0][1] =0;
        wH2[1][0] =0;
        wH2[1][1] =0;
        wH2[2][0] =0;
        wH2[2][1] =0;
        wO[0] =0;
        wO[1] =0;
        wO[2] =0;
        
        int max=3;
        int min=-3;
        //Initialisation of Weights with a random generator
        //Hidden1 Layer
        for(int i=0; i<3; i++)//Index for the "to"(recieving) perceptron
        {
            for(int j=0; j<2; j++)
            {
                int randomInt = randomGenerator.nextInt((max - min) + 1) + min;
                wH1[i][j]=randomInt;
            }
        }
        //Hidden2 Layer
        for(int i=0; i<3; i++)//Index for the "from"(sending) perceptron(including bias)
        {
            for(int j=0; j<2; j++)//Index for the "to"(recieving) perceptron
            {
                int randomInt = randomGenerator.nextInt((max - min) + 1) + min;
                wH2[i][j]=randomInt;
            }
        }
        //Output Layer
        for(int i=0; i<3; i++)//Index for "from"(sending) perceptron(including bias)
        {
            int randomInt = randomGenerator.nextInt((max - min) + 1) + min;
            wO[i]=randomInt;
        }

        //Training the network starts here
        int iteration=0;//Intialisation of iteration count
        double error=0;
        do
        {
            iteration = iteration +1;
            //Feed Forward Neuron Network(First Pass)
            for(int c=0; c<22; c++)//(Considering different pair of inputs(x))
            {   
                //Input Layer Execution
                for(int r=0; r<2; r++)
                {
                    inputOUT [r]=inputIN [r][c];
                }
                inputOUT[2]=bias;
                
                
                //Hidden layer 1 Execution
                double sum=0;
                for(int j=0;j<2;j++)
                {
                    for(int i=0; i<3; i++)
                    {
                        sum=sum+ inputOUT [i]*wH1[i][j];
                    }
                    hidden1IN[j]= sum;
                    //Sigmoid Function
                    hidden1OUT[j]=(double)(1/( 1 + Math.pow(Math.E,(-1*hidden1IN[j]))));
                }
                hidden1OUT[2]=bias;
                
                //Hidden2 Layer Execution
                for(int j=0; j<2; j++)
                {
                    //Weighted Sum of Inputs for each Perceptron(Hidden Layer)
                    sum=0;
                    for(int i=0; i<3; i++)
                    {
                        sum=sum+ hidden1OUT [i]*wH2 [i][j];
                    }
                    hidden2IN[j]= sum;
                    //Sigmoid Function
                    hidden2OUT[j]=(double)(1/( 1 + Math.pow(Math.E,(-1*hidden2IN[j]))));
                    
                }
                hidden2OUT[2]=bias;

                //Output Layer Execution\
                //Weighted Sum of Inputs for the Perceptron(Output Layer)
               sum=0;
                for(int i=0; i<3; i++)
                {
                    sum=sum+ hidden2OUT[i]*wO[i];
                }
                outputIN= sum;
                outputOUT=(double)(1/( 1 + Math.pow(Math.E,(-1*outputIN))));
                //Assign delta for Outputlayer
                double deltaO=outputOUT*(1-outputOUT)*(desired[c]-outputOUT); //Calculating delta
                error=desired[c]-outputOUT;
                
                //Assign delta for nodes in Hidden layer 2
                for(int i=0; i<2; i++)
                {
                    deltaH2[i]=hidden2OUT[i]*(1-hidden2OUT[i])*wO[i]*deltaO;
                }
                
                //Assign delta for nodes in Hidden layer 1
                
                for(int i=0;i<2;i++)
                {
                    sum=0;
                    for(int j=0; j<2; j++)
                    {
                        sum=sum + (wH2[i][j]*deltaH2[j]);
                    }
                    deltaH1[i]=hidden1OUT[i]*(1-hidden1OUT[i])*sum;
                }
                
                //Update new weights(Hidden layer 1)
                double constant=1;
                for(int i=0; i<3; i++)
                {
                    for(int j=0; j<2; j++)
                    {
                        wH1[i][j]=wH1[i][j]+(constant*inputOUT[i]*deltaH1[j]);
                    }
                }
                
               //Update new weights(Hidden layer 2)
               
                for(int i=0; i<3; i++)
                {
                    for(int j=0; j<2; j++)
                    {
                        wH2[i][j]=wH2[i][j]+(constant*inputOUT[i]*deltaH2[j]);
                    }
                }
                //Assign new weights(Output layer)
                for(int i=0; i<3; i++)
                {
                    wO[i] = wO[i]+(constant*deltaO*hidden2OUT[i]);
                }
                /*System.out.println("Feed Forward output is: "+outputOUT);
                System.out.println("Desired output is: "+d[c]);
                System.out.println("error(delta) is: "+delta);*/
                System.out.println("Iteration count="+iteration+";error="+error);
                      
            } 
        }while(iteration<10000);//Number of iterations to be declared

        //TESTING THE NETWORK
System.out.println("TESTING THE NETWORK");
System.out.println("TEST RESULTS:");
        //Test data

        for(int c=0; c<101; c++)
        {
            for(int r=0; r<101; r++)
            {   
                //Input Layer Execution
                inputOUT[0]=(double)(c*0.01);
                inputOUT[1]=(double)(r*0.01);
                inputOUT[2]=bias;

                //Hidden layer 1 Execution
                double sum=0;
                for(int j=0;j<2;j++)
                {
                    for(int i=0; i<3; i++)
                    {
                        sum=sum+ inputOUT [i]*wH1[i][j];
                    }
                    hidden1IN[j]= sum;
                    //Threshold Function
                    if(sum>0){ hidden1OUT[j]=1;}
                    else{hidden1OUT[j]=0;}
                }
                hidden1OUT[2]=bias;
                
                //Hidden2 Layer Execution
                for(int j=0; j<2; j++)
                {
                    //Weighted Sum of Inputs for each Perceptron(Hidden Layer)
                    sum=0;
                    for(int i=0; i<3; i++)
                    {
                        sum=sum+ hidden1OUT [i]*wH2 [i][j];
                    }
                    hidden2IN[j]= sum;
                    //Threshold Function
                    if(sum>0){ hidden2OUT[j]=1;}
                    else{hidden2OUT[j]=0;}
                }
                hidden2OUT[2]=bias;

                //Output Layer Execution\
                //Weighted Sum of Inputs for the Perceptron(Output Layer)
               sum=0;
                for(int i=0; i<3; i++)
                {
                    sum=sum+ hidden2OUT[i]*wO[i];
                }
                outputIN= sum;
                
                //Thresholding for Test Inputs
                int result;
                if(sum>0){ result=1;}
                else     { result=0;}
                
                Output[c][r]=result;
                System.out.println("The Test result for "+"x="+inputOUT[0]+";y="+inputOUT[1]+" is "+result);     
                //ARRAYS FOR PLOTTING THE GRAPH
                Final.add(result);
                FinalInputX.add(inputOUT[0]);
                FinalInputY.add(inputOUT[1]);
            }

            
        }
        
        //ARRAY REARRAGEMENT FOR PLOTTING 
        for(int k=0;k<101;k++){
            for(int i=1;i<102;i++)
                {
                 int j=(i*101)-1-k;
                 FinalResult.add(Final.get(j));
                }
        }     
        
            //Printing the Trained Weights
            System.out.println("trained Weight from Input Node 1 to HiddenLayer1 Node 1 is "+wH1[0][0]);
            System.out.println("trained Weight from Input Node 1 to HiddenLayer1 Node 2 is "+wH1[0][1]);
            System.out.println("trained Weight from Input Node 2 to HiddenLayer1 Node 1 is "+wH1[1][0]);
            System.out.println("trained Weight from Input Node 2 to HiddenLayer1 Node 2 is "+wH1[1][1]);
            System.out.println("Bias Input for HiddenLayer1 Node 1 is "+wH1[2][0]);
            System.out.println("Bias Input for HiddenLayer1 Node 2 is "+wH1[2][1]);
            
            System.out.println("trained Weight from HiddenLayer1 Node 1 to HiddenLayer2 Node 1 is "+wH1[0][0]);
            System.out.println("trained Weight from HiddenLayer1 Node 1 to HiddenLayer2 Node 2 is "+wH1[0][1]);
            System.out.println("trained Weight from HiddenLayer1 Node 2 to HiddenLayer2 Node 1 is "+wH1[1][0]);
            System.out.println("trained Weight from HiddenLayer1 Node 2 to HiddenLayer2 Node 2 is "+wH1[1][1]);
            System.out.println("Bias Input for HiddenLayer2 Node 1 is "+wH2[2][0]);
            System.out.println("Bias Input for HiddenLayer2 Node 2 is "+wH2[2][1]);
            System.out.println("trained Weight from HiddenLayer2 Node 1 to Output Node is "+wO[0]);
            System.out.println("trained Weight from HiddenLayer2 Node 2 to Output Node is "+wO[1]);     
            System.out.println("Bias Input for Output Node is "+wO[2]);
            System.out.println("Yellow Represents Output '0' i.e positive region and Red  Represents Output '1' i.e Negative region");
                        
    Plot_graph();
 }

//Plotiing the graph(X^2 function with inputs with increment 0.01 ranging from 0 to 1 in both x,y axes)
static void Plot_graph()
{
    
    JFrame f= new JFrame();
    JPanel jPanel1 = new JPanel();
    JTextField jTextField1;
    JPanel p = new JPanel();
    jPanel1.setLayout(new GridLayout(101, 101));
    

        for (int i = 0; i < 101 * 101; i++) {
    
            jTextField1 = new JTextField();
            if(FinalResult.get(i)==0)
            jTextField1.setBackground(Color.yellow);//Yellow Represents Output '0'
            else
            jTextField1.setBackground(Color.red);//Red Represents Output '1'
            jPanel1.add(jTextField1);
        }
        f.add(jPanel1);
        f.setSize(800,800);
        f.setVisible(true);
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);       
    }       
}
