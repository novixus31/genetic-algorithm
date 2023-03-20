#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// genotipi nokta vektörü olarak tanımla
typedef vector<Point> Genotype;

// fitness fonksiyonu
double fitness(const Genotype& path, const Mat& elevationMap)
{
    double fitness = 0.0;
    double elevationDiff, slope;
    for(int i=1; i<path.size(); i++)
    {
        // yükseklik farkını hesapla
        elevationDiff = elevationMap.at<int>(path[i]) - elevationMap.at<int>(path[i-1]);
        // eğimi hesapla
        slope = atan(elevationDiff / path[i-1].ddot(path[i-1]) - path[i].ddot(path[i]));
        // verilen formül kullanarak fitness değerini hesapla
        fitness += (abs(0.0 - slope) * 0.5) + (abs(elevationDiff - 0.0) * 0.5);
    }
    return fitness;
}

// genotipin ilk populasyonunu oluştur
vector<Genotype> generateInitialPopulation(int populationSize, const Mat& elevationMap)
{
    vector<Genotype> population;
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> xDist(0, elevationMap.cols - 1);
    uniform_int_distribution<int> yDist(0, elevationMap.rows - 1);
    for(int i=0; i<populationSize; i++)
    {
        Genotype genotype;
        genotype.push_back(Point(392, 244)); // ilk noktayı ekle
        for(int j=1; j<9; j++) // 10 noktalı patika için (ilk ve son noktalar hariç)
        {
            genotype.emplace_back(xDist(gen), yDist(gen));
        }
        genotype.push_back(Point(650, 565)); // son noktayı ekle
        population.push_back(genotype);
    }
    return population;
}

// crossover işlemini gerçekleştir
Genotype crossover(const Genotype& parent1, const Genotype& parent2)
{
    Genotype child;
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(1, parent1.size() - 2); // crossover noktası başlangıç veya son olamaz
    int crossoverPoint = dist(gen);
    for(int i=0; i<crossoverPoint; i++)
    {
        child.push_back(parent1[i]);
    }
    for(int i=crossoverPoint; i<parent2.size(); i++)
    {
        child.push_back(parent2[i]);
    }
    return child;
}

// mutasyon işlemini gerçekleştir
void mutate(Genotype& genotype, const Mat& elevationMap)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(0, elevationMap.cols - 1);
    int mutationPoint = dist(gen);
    genotype[mutationPoint].x = dist(gen);
    genotype[mutationPoint].y = dist(gen);
}

// üreme için en iyi bireyleri seç
vector<Genotype> selection(const vector<Genotype>& population, const Mat& elevationMap)
{
    vector<Genotype> parents;
    for(int i=0; i<2; i++) // select two parents
    {
        double minFitness = numeric_limits<double>::max();
        Genotype bestIndividual;
        for(const auto& individual : population)
        {
            double individualFitness = fitness(individual, elevationMap);
            if(individualFitness < minFitness)
            {
                minFitness = individualFitness;
                bestIndividual = individual;
            }
        }
        parents.push_back(bestIndividual);
    }
    return parents;
}

// main function to perform genetic algorithm
void performGeneticAlgorithm(const Mat& elevationMap)
{
    // define genetic algorithm parameters
    int populationSize = 100;
    int generations = 100;
    double mutationProbability = 0.1;
    // generate initial population
    vector<Genotype> population = generateInitialPopulation(populationSize, elevationMap);

    // run genetic algorithm for given number of generations
    for(int generation=0; generation<generations; generation++)
    {
        // select parents for reproduction
        vector<Genotype> parents = selection(population, elevationMap);

        // create offspring by crossover
        Genotype offspring = crossover(parents[0], parents[1]);

        // apply mutation to offspring with given probability
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<double> dist(0, 1);
        if(dist(gen) < mutationProbability)
        {
            mutate(offspring, elevationMap);
        }

        // calculate fitness of offspring and replace worst individual in population
        double offspringFitness = fitness(offspring, elevationMap);
        double maxFitness = numeric_limits<double>::min();
        int maxFitnessIndex = -1;
        for(int i=0; i<populationSize; i++)
        {
            double individualFitness = fitness(population[i], elevationMap);
            if(individualFitness > maxFitness)
            {
                maxFitness = individualFitness;
                maxFitnessIndex = i;
            }
        }
        if(offspringFitness < maxFitness)
        {
            population[maxFitnessIndex] = offspring;
        }
    }

    // find individual with lowest fitness value as the best solution
    double minFitness = numeric_limits<double>::max();
    Genotype bestIndividual;
    for(const auto& individual : population)
    {
        double individualFitness = fitness(individual, elevationMap);
        if(individualFitness < minFitness)
        {
            minFitness = individualFitness;
            bestIndividual = individual;
        }
    }

    // draw the path on the elevation map
    Mat elevationMapRGB;
    cvtColor(elevationMap, elevationMapRGB, COLOR_GRAY2RGB);
    for(int i=1; i<bestIndividual.size(); i++)
    {
        line(elevationMapRGB, bestIndividual[i-1], bestIndividual[i], Scalar(0,0,255), 1);
    }

    // show the elevation map with the path
    namedWindow("Elevation Map with Path", WINDOW_NORMAL);
    imshow("Elevation Map with Path", elevationMapRGB);
    waitKey(0);
}

//main function
int main()
{
    // load the elevation map
    Mat elevationMap = imread("map.jpeg", IMREAD_GRAYSCALE);

    // perform genetic algorithm to find the path
    performGeneticAlgorithm(elevationMap);

    return 0;
}
