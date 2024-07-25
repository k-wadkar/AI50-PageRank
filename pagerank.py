import os
import random
import re
import sys
from copy import deepcopy

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Contains the set of all pages which the input page links to
    linkedPages = corpus[page]

    # Base probability of any page being the next page assumes the uniform distribution
    baseProbability = (1/len(corpus))
        
    # If the input page does link to other pages...
    if len(linkedPages) != 0:
        # Account for the damping factor in the base probability
        baseProbability *= (1-damping_factor)
        # Additional probability assumes uniform distribution for linked pages, then accounts for damping factor
        additionalProbability = (1/len(linkedPages)) * damping_factor
    # Otherwise if the input page does not link to other pages
    else:
        # ...then the additional probability must be zero 
        additionalProbability = 0

    # Empty dictionary to contain page name against the probability of that page being selected next
    probabilityDistribution = {}

    # For each possible page that could be added to the sample next
    for pages in corpus:
        # Create key-pair entry of page name against base probability
        probabilityDistribution[pages] = baseProbability
        if pages in linkedPages:
            # Add additional probability to base probability if the page happens to be linked by the original page
            probabilityDistribution[pages] += additionalProbability        

    return probabilityDistribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Creates a list containing the name of every page in corpus
    allPages = [page for page in corpus]
    
    # The first sample is selected randomly from the list
    firstSample = random.choice(allPages)

    # List to contain all samples collected (1 -> n samples)
    sample = [deepcopy(firstSample)]
    
    # Previous sample set equal to the first sample
    prevSample = firstSample

    # samples variable used as a 'scapegoat' variable to allow us to collect n samples in total
    for samples in range(1, n):
        # Probability distribution for all possible future samples given the previous sample 
        prevDistribution = transition_model(corpus, prevSample, damping_factor)

        # List containing names of all possible future samples given by prevDistribution
        # (Can't use allPages as was before, since order probably won't be the same as we require it to be which matters later on)
        allPages = [page for page in prevDistribution]
        # List containing the probability for each page being the next sample, in order of appearance in allPages
        probabilities = [prevDistribution[page] for page in prevDistribution]

        # Generates a new sample based on the probability weights
        # (the [0] is there because random.choices returns a list with len 1, rather than just a string)
        newSample = random.choices(allPages, probabilities)[0]

        # Appends the new sample to the overall sample list
        sample.append(deepcopy(newSample))

        # PrevSample updated before the next loop
        prevSample = deepcopy(newSample)

    # Return a dictionary containing each page against its frequency in the sample (i.e. its PageRank)
    return {i: (sample.count(i)/n) for i in set(sample)}


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Contains the initial probability distribution of each page (uniform), equal to 1/number of pages 
    probabilityDistribution = {key: 1/len(corpus) for key in corpus}
    
    # If there is a page which does not have any outward links, give it an outward link to every other page in corpus
    for key in corpus:
        if corpus[key] == set():
            corpus[key] = {key for key in corpus}

    priorDistribution = {}
    while probabilityDistribution != priorDistribution:
        priorDistribution = deepcopy(probabilityDistribution)
        for page in corpus:
            pagesWithInwardLinks = []
            for nestedPage in corpus:
                if page in corpus[nestedPage]:
                    pagesWithInwardLinks.append(nestedPage)
            
            # Summation abstraction
            sum = 0
            for subPage in pagesWithInwardLinks:
                sum += (probabilityDistribution[subPage]/len(corpus[subPage]))

            # List now contains the names of all pages which point to the selected page
            probabilityDistribution[page] = round(
                ((1-damping_factor)/len(corpus))+(damping_factor*sum), 5)
              
    return probabilityDistribution    


if __name__ == "__main__":
    main()
