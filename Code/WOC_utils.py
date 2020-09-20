

# Method to get WOC solution

def get_woc_solution(best_solutions):
    woc_dictionary = []
    for i in range(len(best_solutions[0])):
        woc_dictionary.append(dict(
            (key, [best_solutions[j][i] for j in range(len(best_solutions))].count(key)) for key in
            set([best_solutions[j][i] for j in range(len(best_solutions))])))
    woc_solution = [max(dictionary, key=dictionary.get) for dictionary in woc_dictionary]
    return woc_solution