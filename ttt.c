// Reinforcement learning of a tic tac toe AI
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>

#define BOARD 9

typedef struct
{
    int state[BOARD];
    int act;
    double q;
}q_t;

typedef struct node
{
    q_t q;
    struct node *next;
}node_t;

typedef struct
{
    int mem;
    node_t **nodes;
}hash_t;

typedef struct
{
    float alp;
    float epi;
    float gam;
    hash_t q_val;
}ai_t;

typedef struct 
{
    int board[BOARD];
    int winner;
    int player;
}game_t;

int hash(int state[], int act);
ai_t *initAI();
game_t *initGame();
void printB(int board[]);
char *conv(int board[]);
void play();
int terminal(int board[]);
int *actions(int board[]);
int switchP (game_t *game);
q_t *searchH(int board[], int act, hash_t *H);
q_t *insertH(q_t *q, hash_t *H);
int arrayComp(int a1[], int a2[]);
void arrayDupe(int a1[], int a2[]);
void train(ai_t *AI, int n);
int choose(int board[], int player, ai_t *ai, int random);
double getQ(ai_t *ai, int board[], int action);
void updateQ(ai_t *ai, int state[], int new[], int player, int action, int value);
void freeAI(ai_t *ai);

int main(int argc, char *argv[])
{
    ai_t *AI = initAI();
    if (argc == 1)
    {
        train(AI, 100000);
    }
    else
    {
        int n = atoi(argv[1]);
        if (n)
        {
            train(AI, n);
        }
        else
        {
            printf("Invalid entry");
            exit(EXIT_FAILURE);
        }
    }
    play(AI);
    freeAI(AI);
}

// Compare two array, same 1, different 0.
int arrayComp(int a1[], int a2[])
{
    for (int i = 0; i<BOARD; i++)
    {
        if (a1[i]!=a2[i])
        {
            return 0;
        }
    }
    return 1;
}

// Duplicate array a1 into a2.
void arrayDupe(int a1[], int a2[])
{
    for (int i = 0; i < BOARD; i++)
    {
        a2[i] = a1[i];
    }
}

// Hash the Q value into a key
int hash(int state[], int act)
{
    int result = 0;
    int pos = 1;
    int neg = 1;
    for (int i = 0; i<BOARD; i++)
    {
        if (state[i] > 0)
        {
            pos *= i;
        }
        else if (state[i]<0)
        {
            neg *= i;
        }
    }
    result = act * 100 + pos * 10 + neg;
    return result; 
}

// Initialise the ai
ai_t *initAI()
{
    srand(time(NULL));
    ai_t *ai = malloc(sizeof(ai_t));
    assert(ai);
    ai->alp = 0.8;
    ai->epi = 0.2;
    ai->gam = 0.9;
    int size = 1000;
    (ai->q_val).mem = size;
    (ai->q_val).nodes = malloc(size * sizeof(node_t *));
    assert((ai->q_val).nodes);
    for (int i = 0; i<size; i++)
    {
        ai->q_val.nodes[i] = NULL;
    }
    return ai;
}

// Insert a Q value.
q_t *insertH(q_t *q, hash_t *H)
{
    int key = hash(q->state, q->act);
    node_t *node= malloc(sizeof(node_t));
    assert(node);
    node->q = *q;
    node->next = NULL;
    if (key >= H->mem)
    {
        int pastM = H->mem;
        H->mem = key+1;
        H->nodes = realloc(H->nodes, H->mem * sizeof(node_t *));
        assert(H->nodes);
        for(int i = pastM; i < H->mem; i++)
        {
            H->nodes[i] = NULL;
        }
    }
    if (H->nodes[key] == NULL)
    {
        H->nodes[key] = node;
    }   
    else
    {
        node_t *current = H->nodes[key];
        while(current->next != NULL)
        {
            current = current->next;
        }
        current->next = node;
    } 
    free(q);
    q = &(node->q);
    return q;
}

// Search for the Q value.
q_t *searchH(int board[], int act, hash_t *H)
{
    int key = hash(board, act);
    if (H->mem <= key)
    {
        return NULL;
    }
    else
    {      
        node_t *current = H->nodes[key];
        while(current != NULL)
        {
            if ((current->q).act == act && arrayComp((current->q).state, board))
            {
                return &(current->q);
            }
            current = current->next;
        }
        return NULL;
    }
}

// Initialise the game, player -1 starts the game
game_t *initGame()
{
    game_t *game = malloc(sizeof(game_t));
    game->winner = 0;
    game->player = 1;
    for (int i = 0; i<BOARD; i++)
    {
        game->board[i] = 0;
    }
    return game;
}

// Print the board
void printB(int board[])
{
    char *text = conv(board);
    printf("-------------\n");
    for (int i = 0; i<3; i++)
    {
        printf("|");
        for (int j = 0; j < 3; j++)
        {
            printf(" %c |", text[3*i+j]);
        }
        printf("\n-------------\n");
    }
}

// Convert the board from number to text
char *conv(int board[])
{
    int p;
    char *result = malloc(10 * sizeof(char));
    result[9] = '\0';
    for (int i = 0; i<BOARD; i++)
    {
        p = board[i];
        if (p == 0)
        {
            result[i] = i+'1';
        }
        else if (p == 1)
        {
            result[i] = 'O';
        }
        else
        {
            result[i] = 'X';
        }
    }
    return result;
}

// Check whether the game has end and return the winner;
int terminal(int board[])
{
    int found = -1;
    for (int i = 0; i<3; i++)
    {
        if (board[i] != 0 && board[i] == board[i+3] && board[i] == board[i+6])
        {
            found = i;
        }
        else if (board[i*3] != 0 && board[i*3] == board[i*3+1] && board[i*3] == board[i*3+2])
        {
            found = i*3;
        }
    }
    if (board[4] != 0 && ((board[4] == board[0] && board[4] == board[8]) || (board[4] == board[2] && board[4] == board[6])))
    {
        found = 4;
    }
    if (found >= 0)
    {
        if (board[found] == -1)
        {
            return -1;
        }
        else
        {
            return 1;
        }
    }
    found = 2;
    for (int i = 0; i < BOARD; i++)
    {
        if (board[i] == 0)
        {
            found = 0;
        }
    }
    return found;
}

// Switch the player and return the current player
int switchP (game_t *game)
{
    if (game->player < 0)
    {
        game->player = 1;
        return 2;
    }
    else
    {
        game->player = -1;
        return 1;
    }
}

// Return an array of available actions. The first element is the number of moves
int *actions(int board[])
{
    int *valid = malloc((BOARD+2)*sizeof(int));
    int count = 1;
    int i;
    for(i = 0; i<BOARD; i++)
    {
        if (board[i] == 0)
        {
            valid[count] = i;
            count++;
        }
    }
    valid[count] = -1;
    valid[0] = count - 1;
    return valid;
}

// Play tic tac toe
void play(ai_t *ai)
{
    game_t *game = initGame();
    int move;
    int aiP = rand()%2;
    int *valid;
    printf("AI is player %d\n", aiP+1);
    if (aiP == 0)
    {
        aiP = -1;
    }
    printB(game->board);
    while(terminal(game->board)==0)
    {
        printf("Player %d's turn: ", switchP(game));
        if (game->player == aiP)
        {
            move = choose(game->board, aiP, ai, 0);
            printf("%d\n", move+1);
        }
        else
        {
            valid = actions(game->board);
            int correct = 0;
            while (correct == 0)
            {
                scanf("%d", &move);
                move --;
                for (int i = 1; valid[i]!= -1; i++)
                {
                    if (move == valid[i])
                    {
                        correct = 1;
                        break;
                    }
                }
                if (!correct)
                {
                    printf("Invalid entry, enter another value: ");
                }
            }
        }
        (game->board)[move] = game->player;
        printB(game->board);
    }
    if (terminal(game->board)==2)
    {
        printf("Game end: Draw!");
    }
    else
    {
        switchP(game);
        printf("Game end: player %d wins!", switchP(game));
    }
    free(valid);
    free(game);
}

// Train the ai
void train(ai_t *ai, int n)
{
    game_t *game;
    int move;
    int last[BOARD];
    int move5;
    for (int i = 0; i<n; i++)
    {
        game = initGame();
        move = -1;
        while(terminal(game->board) == 0)
        {
            if (move != -1)
            {
                updateQ(ai, last, game->board, game->player, move, 0);
            }
            switchP(game);
            move = choose(game->board, game->player, ai, 1);
            arrayDupe(game->board, last);
            game->board[move] = game->player;
        }
        updateQ(ai, last, game->board, game->player, move, terminal(game->board));
        free(game);
    }
}

// Chose a best move
int choose(int board[], int player, ai_t *ai, int random)
{
    double epi = ai->epi;
    if (!random)
    {
        epi = 0;
    }
    int *possible = actions(board);
    if ((double)rand()/RAND_MAX < epi)
    {
        return possible[(rand()%possible[0])+1];
    }
    double best = -2;
    int move = -1;
    double Q;
    for (int i = 1; possible[i]!=-1; i++)
    {
        Q = getQ(ai, board, possible[i]) * player;
        if (Q > best)
        {
            best = Q;
            move = possible[i];
        }
    }
    return move;
}

// Get the Q value of a move
double getQ(ai_t *ai, int board[], int action)
{
    q_t *q = searchH(board, action, &(ai->q_val));
    if (q == NULL)
    {
        return 0;
    }
    return q->q;
}

// Update the Q value, player is the player that makes the move
void updateQ(ai_t *ai, int state[], int new[], int player, int action, int value)
{
    if (value == 2)
    {
        value = 0;
    }
    q_t *q = searchH(state,action, &(ai->q_val));
    if (q == NULL)
    {
        q = malloc(sizeof(q_t));
        assert(q);
        arrayDupe(state, q->state);
        q->act = action;
        q->q = 0;
        q = insertH(q, &(ai->q_val));
    }
    double future = 0;
    int move = -1;
    if (!terminal(new))
    {
        move = choose(new, player*-1, ai, 0);
        future = getQ(ai, new, move);
    }
    double result = q->q + ai->alp*(value + ai->gam*future - q->q);
    q->q = result;
    /*
    if (move == -1 && terminal(new) != 2)
    {
        printf("Action: %d; Next move: %d; Terminal: %d\n", action, move, terminal(new));
        printf("Current: %.2f; Value: %.2f; Future: %.2f; Result: %.2f\n", q->q, (double)value, future, result);
        printf("Check: %.2f\n", getQ(ai, state,action));
    }
    */
}

// Free the space allocated to ai
void freeAI(ai_t *ai)
{
    for (int i = 0; i < ai->q_val.mem; i++)
    {
        free(ai->q_val.nodes[i]);
        ai->q_val.nodes[i] = NULL;
    }
    free(ai->q_val.nodes);
    free(ai);
}