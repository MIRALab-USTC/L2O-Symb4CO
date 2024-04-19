#include <math.h>
#include "scip/branch_symb_local.h"
#include <stdio.h>
#include <stdlib.h>
#include "scip/scip.h"
#include "scip/struct_lp.h"
#include <time.h>

typedef float Symb_Feature[NUM_FEATURE];

float ScoreFunc(Symb_Feature feature);

SCIP_RETCODE symbAllocSetFeatures(Symb_Feature** allfeatures_ptr, SCIP_VAR*** lpcands_ptr, int* nlpcands_ptr, SCIP* scip, SCIP_BRANCHRULE* branchrule)
{
    SCIP_VAR** lpcands;
    // SCIP_Real *lpcandssol, *lpcandsfrac;
    int nlpcands;
    Symb_Feature* allfeatures;

    // SCIP_CALL( SCIPgetLPBranchCands(scip, &lpcands, &lpcandssol, &lpcandsfrac, NULL, &nlpcands, NULL) );
    SCIPgetPseudoBranchCands(scip, &lpcands, &nlpcands, NULL);
    if(nlpcands_ptr != NULL) *nlpcands_ptr = nlpcands;
    if(lpcands_ptr != NULL) *lpcands_ptr = lpcands;

    SCIP_BRANCHRULEDATA* branchruledata;
    branchruledata = SCIPbranchruleGetData(branchrule);
    SYMB_ROOTINFO* rootinfo = &(branchruledata->rootinfo);

    if(nlpcands > 0)
    {
        SCIP_ALLOC( BMSallocMemorySize(allfeatures_ptr, sizeof(Symb_Feature) * nlpcands ) );
        allfeatures = *allfeatures_ptr;
    }
    else
    {
        *allfeatures_ptr = NULL;
        allfeatures=NULL;
    }

    if(branchruledata->inited) ;
    else
    {
        branchruledata->inited=TRUE;
        #ifdef USE_ROOT
            SCIP_BRANCHRULEDATA* branchruledata;
            branchruledata = SCIPbranchruleGetData(branchrule);
            SYMB_ROOTINFO* rootinfo = &(branchruledata->rootinfo);

            SCIP_COL** cols = SCIPgetLPCols(scip);
            int ncols = SCIPgetNLPCols(scip);

            int max_len = -1;
            for(int i=0; i < ncols; ++i)
            {
                max_len = MAX(SCIPcolGetIndex(cols[i]), max_len);
            }
            ++max_len;



            #ifdef USE_TYPE
            #ifdef TYPE_0
            if(max_len > 0) SCIP_ALLOC( BMSallocMemorySize(&(rootinfo->type_0), max_len * sizeof(float)) );
            #endif
            #ifdef TYPE_1
            if(max_len > 0) SCIP_ALLOC( BMSallocMemorySize(&(rootinfo->type_1), max_len * sizeof(float)) );
            #endif
            #ifdef TYPE_2
            if(max_len > 0) SCIP_ALLOC( BMSallocMemorySize(&(rootinfo->type_2), max_len * sizeof(float)) );
            #endif
            #ifdef TYPE_3
            if(max_len > 0) SCIP_ALLOC( BMSallocMemorySize(&(rootinfo->type_3), max_len * sizeof(float)) );
            #endif
            #endif

            #ifdef USE_COEFS
            if(max_len > 0) SCIP_ALLOC( BMSallocMemorySize(&(rootinfo->coefs), max_len * sizeof(float)) );
            #endif
            #ifdef COEFS_POS
            if(max_len > 0) SCIP_ALLOC( BMSallocMemorySize(&(rootinfo->coefs_pos), max_len * sizeof(float)) );
            #endif
            #ifdef COEFS_NEG
            if(max_len > 0) SCIP_ALLOC( BMSallocMemorySize(&(rootinfo->coefs_neg), max_len * sizeof(float)) );
            #endif
            #ifdef NNZRS
            if(max_len > 0) SCIP_ALLOC( BMSallocMemorySize(&(rootinfo->nnzrs), max_len * sizeof(float)) );
            #endif
            #ifdef USE_ROOT_CDEG_MEAN
            if(max_len > 0) SCIP_ALLOC( BMSallocMemorySize(&(rootinfo->root_cdeg_mean), max_len * sizeof(float)) );
            #endif
            #ifdef ROOT_CDEG_VAR
            if(max_len > 0) SCIP_ALLOC( BMSallocMemorySize(&(rootinfo->root_cdeg_var), max_len * sizeof(float)) );
            #endif
            #ifdef USE_ROOT_CDEG_MIN
            if(max_len > 0) SCIP_ALLOC( BMSallocMemorySize(&(rootinfo->root_cdeg_min), max_len * sizeof(float)) );
            #endif
            #ifdef USE_ROOT_CDEG_MAX
            if(max_len > 0) SCIP_ALLOC( BMSallocMemorySize(&(rootinfo->root_cdeg_max), max_len * sizeof(float)) );
            #endif
            #ifdef ROOT_PCOEFS_COUNT
            if(max_len > 0) SCIP_ALLOC( BMSallocMemorySize(&(rootinfo->root_pcoefs_count), max_len * sizeof(float)) );
            #endif
            #ifdef ROOT_PCOEFS_MEAN
            if(max_len > 0) SCIP_ALLOC( BMSallocMemorySize(&(rootinfo->root_pcoefs_mean), max_len * sizeof(float)) );
            #endif
            #ifdef ROOT_PCOEFS_VAR
            if(max_len > 0) SCIP_ALLOC( BMSallocMemorySize(&(rootinfo->root_pcoefs_var), max_len * sizeof(float)) );
            #endif
            #ifdef ROOT_PCOEFS_MIN
            if(max_len > 0) SCIP_ALLOC( BMSallocMemorySize(&(rootinfo->root_pcoefs_min), max_len * sizeof(float)) );
            #endif
            #ifdef ROOT_PCOEFS_MAX
            if(max_len > 0) SCIP_ALLOC( BMSallocMemorySize(&(rootinfo->root_pcoefs_max), max_len * sizeof(float)) );
            #endif
            #ifdef ROOT_NCOEFS_COUNT
            if(max_len > 0) SCIP_ALLOC( BMSallocMemorySize(&(rootinfo->root_ncoefs_count), max_len * sizeof(float)) );
            #endif
            #ifdef ROOT_NCOEFS_MEAN
            if(max_len > 0) SCIP_ALLOC( BMSallocMemorySize(&(rootinfo->root_ncoefs_mean), max_len * sizeof(float)) );
            #endif
            #ifdef ROOT_NCOEFS_VAR
            if(max_len > 0) SCIP_ALLOC( BMSallocMemorySize(&(rootinfo->root_ncoefs_var), max_len * sizeof(float)) );
            #endif
            #ifdef ROOT_NCOEFS_MIN
            if(max_len > 0) SCIP_ALLOC( BMSallocMemorySize(&(rootinfo->root_ncoefs_min), max_len * sizeof(float)) );
            #endif
            #ifdef ROOT_NCOEFS_MAX
            if(max_len > 0) SCIP_ALLOC( BMSallocMemorySize(&(rootinfo->root_ncoefs_max), max_len * sizeof(float)) );
            #endif

            for(int i=0; i < ncols; ++i)
            {
                SCIP_COL* col = cols[i];
                int col_i = SCIPcolGetIndex(col);
                assert(col_i < max_len);
                SCIP_ROW** neighbors = SCIPcolGetRows(col);
                int nb_neighbors = SCIPcolGetNNonz(col);
                SCIP_Real* nonzero_coefs_raw = SCIPcolGetVals(col);

                #ifdef USE_TYPE
                SCIP_VAR* var = SCIPcolGetVar(col);
                SCIP_VARTYPE vartype = SCIPvarGetType(var);
                #ifdef TYPE_0
                rootinfo->type_0[col_i] = (float) (vartype==SCIP_VARTYPE_BINARY);
                #endif
                #ifdef TYPE_1
                rootinfo->type_1[col_i] = (float) (vartype==SCIP_VARTYPE_INTEGER);
                #endif
                #ifdef TYPE_2
                rootinfo->type_2[col_i] = (float) (vartype==SCIP_VARTYPE_IMPLINT);
                #endif
                #ifdef TYPE_3
                rootinfo->type_3[col_i] = (float) (vartype==SCIP_VARTYPE_CONTINUOUS);
                #endif
                #endif

                SCIP_Real obj_value = SCIPcolGetObj(col);
                #ifdef USE_COEFS
                rootinfo->coefs[col_i] = (float) obj_value;
                #endif
                #ifdef COEFS_POS
                rootinfo->coefs_pos[col_i] = MAX(obj_value, 0.);
                #endif
                #ifdef COEFS_NEG
                rootinfo->coefs_neg[col_i] = MIN(obj_value, 0.);
                #endif
                #ifdef NNZRS
                rootinfo->nnzrs[col_i] = (float) nb_neighbors;
                #endif

                #ifdef USE_ROOT_COLUMN
                float cdeg_var=0., cdeg_mean=0., cdeg_min=INFINITY, cdeg_max=-INFINITY, pcoefs_var=0., pcoefs_mean=0., pcoefs_min=INFINITY, pcoefs_max=-INFINITY, ncoefs_var=0., ncoefs_mean=0., ncoefs_min=INFINITY, ncoefs_max=-INFINITY;
                int pcoefs_count=0, ncoefs_count=0;
                float coef, cdeg;
                for(int neighbor_index=0; neighbor_index < nb_neighbors; ++neighbor_index)
                {
                    #ifdef USE_ROOT_CDEG
                    cdeg = (float) SCIProwGetNNonz(neighbors[neighbor_index]);
                    #ifdef USE_ROOT_CDEG_MEAN
                    cdeg_mean += cdeg;
                    #endif
                    #ifdef ROOT_CDEG_VAR
                    cdeg_var += cdeg * cdeg;
                    #endif
                    #ifdef USE_ROOT_CDEG_MAX
                    cdeg_max = MAX(cdeg_max, cdeg);
                    #endif
                    #ifdef USE_ROOT_CDEG_MIN
                    cdeg_min = MIN(cdeg_min, cdeg);
                    #endif
                    #endif

                    coef = (float) nonzero_coefs_raw[neighbor_index];
                    if(coef > 0)
                    {   
                        #ifdef USE_ROOT_COLUMN_PCOEFS
                        ++pcoefs_count;
                        #ifdef USE_ROOT_PCOEFS_MEAN
                        pcoefs_mean += coef;
                        #endif
                        #ifdef ROOT_PCOEFS_VAR
                        pcoefs_var += coef * coef;
                        #endif
                        #ifdef ROOT_PCOEFS_MIN
                        pcoefs_min = MIN(pcoefs_min, coef);
                        #endif
                        #ifdef ROOT_PCOEFS_MAX
                        pcoefs_max = MAX(pcoefs_max, coef);
                        #endif
                        #endif
                    }
                    else if(coef < 0)
                    {
                        #ifdef USE_ROOT_COLUMN_NCOEFS
                        ++ncoefs_count;
                        #ifdef USE_ROOT_NCOEFS_MEAN
                        ncoefs_mean += coef;
                        #endif
                        #ifdef ROOT_NCOEFS_VAR
                        ncoefs_var += coef * coef;
                        #endif
                        #ifdef ROOT_NCOEFS_MIN
                        ncoefs_min = MIN(ncoefs_min, coef);
                        #endif
                        #ifdef ROOT_NCOEFS_MAX
                        ncoefs_max = MAX(ncoefs_max, coef);
                        #endif
                        #endif
                    }
                }
                if(nb_neighbors>0)
                {
                    #ifdef USE_ROOT_CDEG_MEAN
                    cdeg_mean /= nb_neighbors;
                    #endif
                    #ifdef ROOT_CDEG_VAR
                    cdeg_var = cdeg_var / nb_neighbors - cdeg_mean*cdeg_mean;
                    #endif
                }
                else cdeg_max = cdeg_min = 0.;

                #ifdef USE_ROOT_CDEG_MEAN
                rootinfo->root_cdeg_mean[col_i] = cdeg_mean; 
                #endif
                #ifdef ROOT_CDEG_VAR
                rootinfo->root_cdeg_var[col_i] = cdeg_var;
                #endif
                #ifdef USE_ROOT_CDEG_MAX
                rootinfo->root_cdeg_max[col_i] = cdeg_max;
                #endif
                #ifdef USE_ROOT_CDEG_MIN
                rootinfo->root_cdeg_min[col_i] = cdeg_min;
                #endif

                if(pcoefs_count > 0)
                {
                    #ifdef USE_ROOT_PCOEFS_MEAN
                    pcoefs_mean /= pcoefs_count;
                    #endif
                    #ifdef ROOT_PCOEFS_VAR
                    pcoefs_var = pcoefs_var / pcoefs_count - pcoefs_mean * pcoefs_mean;
                    #endif
                }
                else pcoefs_min = pcoefs_max = 0.;

                #ifdef USE_ROOT_PCOEFS_MEAN
                rootinfo->root_pcoefs_mean[col_i] = pcoefs_mean;
                #endif
                #ifdef ROOT_PCOEFS_VAR
                rootinfo->root_pcoefs_var[col_i] = pcoefs_var;
                #endif
                #ifdef ROOT_PCOEFS_COUNT
                rootinfo->root_pcoefs_count[col_i] = (float) pcoefs_count;
                #endif
                #ifdef ROOT_PCOEFS_MAX
                rootinfo->root_pcoefs_max[col_i] = pcoefs_max;
                #endif
                #ifdef ROOT_PCOEFS_MIN
                rootinfo->root_pcoefs_min[col_i] = pcoefs_min;
                #endif

                if(ncoefs_count > 0)
                {
                    #ifdef USE_ROOT_NCOEFS_MEAN
                    ncoefs_mean /= ncoefs_count;
                    #endif
                    #ifdef ROOT_NCOEFS_VAR
                    ncoefs_var = ncoefs_var / ncoefs_count - ncoefs_mean * ncoefs_mean;
                    #endif
                }
                else ncoefs_min = ncoefs_max = 0.;

                #ifdef USE_ROOT_NCOEFS_MEAN
                rootinfo->root_ncoefs_mean[col_i] = ncoefs_mean;
                #endif
                #ifdef ROOT_NCOEFS_VAR
                rootinfo->root_ncoefs_var[col_i] = ncoefs_var;
                #endif
                #ifdef ROOT_NCOEFS_COUNT
                rootinfo->root_ncoefs_count[col_i] = (float) ncoefs_count;
                #endif
                #ifdef ROOT_NCOEFS_MAX
                rootinfo->root_ncoefs_max[col_i] = ncoefs_max;
                #endif
                #ifdef ROOT_NCOEFS_MIN
                rootinfo->root_ncoefs_min[col_i] = ncoefs_min;
                #endif



                #endif
            }

        #endif
    }

    #ifdef USE_ACTIVE
    // float* act_cons_w1, act_cons_w2, act_cons_w3, act_cons_w4;
    int nrows = SCIPgetNLPRows(scip);
    SCIP_ROW** rows = SCIPgetLPRows(scip);
    // float constraint_sum, abs_coef;
    // int neighbor_var_index;
    #ifdef USE_ACTIVE1
    float* act_cons_w1;
    SCIP_ALLOC( BMSallocMemorySize(&(act_cons_w1), sizeof(float) * nrows ) );
    #endif
    #ifdef USE_ACTIVE2
    float* act_cons_w2;
    SCIP_ALLOC( BMSallocMemorySize(&(act_cons_w2), sizeof(float) * nrows ) );
    #endif
    #ifdef USE_ACTIVE3
    float* act_cons_w3;
    SCIP_ALLOC( BMSallocMemorySize(&(act_cons_w3), sizeof(float) * nrows ) );
    #endif
    #ifdef USE_ACTIVE4
    float* act_cons_w4;
    SCIP_ALLOC( BMSallocMemorySize(&(act_cons_w4), sizeof(float) * nrows ) );
    #endif
    for(int row_index=0; row_index < nrows; ++row_index)
    {
        SCIP_ROW* row = rows[row_index];
        float rhs = (float) SCIProwGetRhs(row);
        float lhs = (float) SCIProwGetLhs(row);
        float activity = (float) SCIPgetRowActivity(scip, row);
        if( SCIPisEQ(scip, activity, rhs) || SCIPisEQ(scip, activity, lhs) )
        {
            SCIP_COL** neighbor_columns = SCIProwGetCols(row);
            int neighbor_ncolumns = SCIProwGetNNonz(row);
            SCIP_Real* neighbor_columns_values = SCIProwGetVals(row);

            #ifdef USE_ACTIVE1
            act_cons_w1[row_index] = 1;
            #endif
            #ifdef USE_ACTIVE4
            act_cons_w4[row_index] = REALABS(SCIProwGetDualsol(row));
            #endif

            
            float constraint_sum_2 = 0., constraint_sum_3=0.;
            for(int neighbor_column_index=0; neighbor_column_index < neighbor_ncolumns; ++neighbor_column_index)
            {
                #ifdef USE_ACTIVE2
                constraint_sum_2 += REALABS(neighbor_columns_values[neighbor_column_index]);
                #endif
                #ifdef USE_ACTIVE3
                SCIP_VAR* neighbor_var = SCIPcolGetVar(neighbor_columns[neighbor_column_index]);
                int neighbor_var_index = SCIPvarGetIndex(neighbor_var);
                for(int cand_i=0; cand_i < nlpcands; ++cand_i)
                {
                    SCIP_VAR* var = lpcands[cand_i];
                    if(SCIPvarGetIndex(var) == neighbor_var_index)
                    {
                        constraint_sum_3 += REALABS(neighbor_columns_values[neighbor_column_index]);
                        break;
                    }
                }
                #endif
            }
            #ifdef USE_ACTIVE2
            // act_cons_w2[row_index] = 1 if constraint_sum == 0 else 1 / constraint_sum
            act_cons_w2[row_index] = (constraint_sum_2 > 0) ? (1 / constraint_sum_2) : (1.);
            #endif
            #ifdef USE_ACTIVE3
            act_cons_w3[row_index] = (constraint_sum_3 > 0) ? (1 / constraint_sum_3) : (1.);
            #endif
        }
    }
    #endif

    #ifdef INC_VAL
    SCIP_SOL* sol = SCIPgetBestSol(scip);
    #endif
    #ifdef AGE
    float age_norm = (float) (SCIPgetNLPs(scip)+5);
    #endif

    for(int c = 0; c < nlpcands; ++c )
    {
        // var = (<Variable>candidates[cand_i]).scip_var
        SCIP_VAR* var = lpcands[c];
        SCIP_COL* col = SCIPvarGetCol(var);
        int col_i = SCIPcolGetIndex(col);
        SCIP_ROW** neighbors = SCIPcolGetRows(col);
        int nb_neighbors = SCIPcolGetNNonz(col);
        SCIP_Real* nonzero_coefs_raw = SCIPcolGetVals(col);

        float solval = SCIPcolGetPrimsol(col);
        float solfrac = SCIPfeasFrac(scip, solval);
        // float lpfrac = (float) lpcandsfrac[c];

        #ifdef USE_ROOT

        #ifdef USE_TYPE
        #ifdef TYPE_0
        allfeatures[c][TYPE_0] = rootinfo->type_0[col_i];
        #endif
        #ifdef TYPE_1
        allfeatures[c][TYPE_1] = rootinfo->type_1[col_i];
        #endif
        #ifdef TYPE_2
        allfeatures[c][TYPE_2] = rootinfo->type_2[col_i];
        #endif
        #ifdef TYPE_3
        allfeatures[c][TYPE_3] = rootinfo->type_3[col_i];
        #endif
        #endif

        #ifdef COEF_NORMALIZED
        allfeatures[c][COEF_NORMALIZED] = rootinfo->coefs[col_i];
        #endif

        #ifdef COEFS
        allfeatures[c][COEFS] = rootinfo->coefs[col_i];
        #endif
        #ifdef COEFS_POS
        allfeatures[c][COEFS_POS] = rootinfo->coefs_pos[col_i];
        #endif
        #ifdef COEFS_NEG
        allfeatures[c][COEFS_NEG] = rootinfo->coefs_neg[col_i];
        #endif
        #ifdef NNZRS
        allfeatures[c][NNZRS] = rootinfo->nnzrs[col_i];
        #endif
        #ifdef ROOT_CDEG_MEAN
        allfeatures[c][ROOT_CDEG_MEAN] = rootinfo->root_cdeg_mean[col_i];
        #endif
        #ifdef ROOT_CDEG_VAR
        allfeatures[c][ROOT_CDEG_VAR] = rootinfo->root_cdeg_var[col_i];
        #endif
        #ifdef ROOT_CDEG_MIN
        allfeatures[c][ROOT_CDEG_MIN] = rootinfo->root_cdeg_min[col_i];
        #endif
        #ifdef ROOT_CDEG_MAX
        allfeatures[c][ROOT_CDEG_MAX] = rootinfo->root_cdeg_max[col_i];
        #endif
        #ifdef ROOT_PCOEFS_COUNT
        allfeatures[c][ROOT_PCOEFS_COUNT] = rootinfo->root_pcoefs_count[col_i];
        #endif
        #ifdef ROOT_PCOEFS_MEAN
        allfeatures[c][ROOT_PCOEFS_MEAN] = rootinfo->root_pcoefs_mean[col_i];
        #endif
        #ifdef ROOT_PCOEFS_VAR
        allfeatures[c][ROOT_PCOEFS_VAR] = rootinfo->root_pcoefs_var[col_i];
        #endif
        #ifdef ROOT_PCOEFS_MIN
        allfeatures[c][ROOT_PCOEFS_MIN] = rootinfo->root_pcoefs_min[col_i];
        #endif
        #ifdef ROOT_PCOEFS_MAX
        allfeatures[c][ROOT_PCOEFS_MAX] = rootinfo->root_pcoefs_max[col_i];
        #endif
        #ifdef ROOT_NCOEFS_COUNT
        allfeatures[c][ROOT_NCOEFS_COUNT] = rootinfo->root_ncoefs_count[col_i];
        #endif
        #ifdef ROOT_NCOEFS_MEAN
        allfeatures[c][ROOT_NCOEFS_MEAN] = rootinfo->root_ncoefs_mean[col_i];
        #endif
        #ifdef ROOT_NCOEFS_VAR
        allfeatures[c][ROOT_NCOEFS_VAR] = rootinfo->root_ncoefs_var[col_i];
        #endif
        #ifdef ROOT_NCOEFS_MIN
        allfeatures[c][ROOT_NCOEFS_MIN] = rootinfo->root_ncoefs_min[col_i];
        #endif
        #ifdef ROOT_NCOEFS_MAX
        allfeatures[c][ROOT_NCOEFS_MAX] = rootinfo->root_ncoefs_max[col_i];
        #endif
        #endif

        #ifdef USE_LB
        float lb = SCIPcolGetLb(col);
        #ifdef HAS_LB
        allfeatures[c][HAS_LB] = (float) ( !SCIPisInfinity(scip, REALABS(lb)) );
        #endif
        #ifdef SOL_IS_AT_LB
        allfeatures[c][SOL_IS_AT_LB] = (float) SCIPisEQ(scip, solval, lb);
        #endif
        #endif

        #ifdef USE_UB
        float ub = SCIPcolGetUb(col);
        #ifdef HAS_UB
        allfeatures[c][HAS_UB] = (float) ( !SCIPisInfinity(scip, REALABS(ub)) );
        #endif
        #ifdef SOL_IS_AT_UB
        allfeatures[c][SOL_IS_AT_UB] = (float) SCIPisEQ(scip, solval, ub);
        #endif
        #endif

        #ifdef USE_BASIS_STATUS
        SCIP_BASESTAT basis_status = SCIPcolGetBasisStatus(col);
        #ifdef BASIS_STATUS_0
        allfeatures[c][BASIS_STATUS_0] = (float) (basis_status == SCIP_BASESTAT_LOWER);
        #endif
        #ifdef BASIS_STATUS_1
        allfeatures[c][BASIS_STATUS_1] = (float) (basis_status == SCIP_BASESTAT_BASIC);
        #endif
        #ifdef BASIS_STATUS_2
        allfeatures[c][BASIS_STATUS_2] = (float) (basis_status == SCIP_BASESTAT_UPPER);
        #endif
        #ifdef BASIS_STATUS_3
        allfeatures[c][BASIS_STATUS_3] = (float) (basis_status == SCIP_BASESTAT_ZERO);
        #endif
        #endif

        #ifdef REDUCED_COST
        allfeatures[c][REDUCED_COST] = SCIPgetColRedcost(scip, col);
        #endif

        #ifdef AGE
        allfeatures[c][AGE] = ((float) col->age) / age_norm;
        #endif

        #ifdef SOL_VAL
        allfeatures[c][SOL_VAL] = solval;
        #endif
        #ifdef SOL_FRAC
        allfeatures[c][SOL_FRAC] = SCIPfeasFrac(scip, solval);
        #endif

        #ifdef INC_VAL
        allfeatures[c][INC_VAL] = SCIPgetSolVal(scip, sol, var);
        #endif
        #ifdef AVG_INC_VAL
        allfeatures[c][AVG_INC_VAL] = SCIPvarGetAvgSol(var);
        #endif

        #ifdef SOLFRACS
        allfeatures[c][SOLFRACS] = solfrac;
        #endif
        #ifdef SLACK
        allfeatures[c][SLACK] = MIN(solfrac, 1.-solfrac);
        #endif

        #ifdef USE_PSEUDO
        float ps_up = (float) SCIPgetVarPseudocost(scip, var, SCIP_BRANCHDIR_UPWARDS), ps_down = (float) SCIPgetVarPseudocost(scip, var, SCIP_BRANCHDIR_DOWNWARDS);
        #ifdef PS_UP
        allfeatures[c][PS_UP] = ps_up;
        #endif
        #ifdef PS_DOWN
        allfeatures[c][PS_DOWN] = ps_down;
        #endif
        #ifdef PS_SUM
        allfeatures[c][PS_SUM] = ps_up + ps_down;
        #endif
        #ifdef PS_RATIO
        allfeatures[c][PS_RATIO] = (ps_up > 0.) ? (ps_up / (ps_up + ps_down)) : 0.;
        #endif
        #ifdef PS_PRODUCT
        allfeatures[c][PS_PRODUCT] = ps_up * ps_down;
        #endif
        #endif

        #ifdef NB_UP_INFEAS
        allfeatures[c][NB_UP_INFEAS] = (float) SCIPvarGetCutoffSum(var, SCIP_BRANCHDIR_UPWARDS);
        #endif
        #ifdef NB_DOWN_INFEAS
        allfeatures[c][NB_DOWN_INFEAS] = (float) SCIPvarGetCutoffSum(var, SCIP_BRANCHDIR_DOWNWARDS);
        #endif
        SCIP_Longint nbranchings;
        #ifdef FRAC_UP_INFEAS
        nbranchings = SCIPvarGetNBranchings(var, SCIP_BRANCHDIR_UPWARDS);
        allfeatures[c][FRAC_UP_INFEAS] = (nbranchings > 0)?((float) SCIPvarGetCutoffSum(var, SCIP_BRANCHDIR_UPWARDS) / nbranchings):(0.);
        #endif
        #ifdef FRAC_DOWN_INFEAS
        nbranchings = SCIPvarGetNBranchings(var, SCIP_BRANCHDIR_DOWNWARDS);
        allfeatures[c][FRAC_DOWN_INFEAS] = (nbranchings > 0)?((float) SCIPvarGetCutoffSum(var, SCIP_BRANCHDIR_DOWNWARDS) / nbranchings):(0.);
        #endif

        #ifdef USE_COLUMN
        float coef, lhs, rhs;
        #ifdef USE_CDEG
        float cdeg_var=0., cdeg_mean=0., cdeg_min=INFINITY, cdeg_max=-INFINITY;
        #endif
        #ifdef USE_RHS
        float prhs_ratio_max=-1., prhs_ratio_min = 1., nrhs_ratio_max = -1., nrhs_ratio_min = 1.;
        #endif
        #ifdef USE_OTA
        float ota_pp_max=0., ota_pp_min=1., ota_pn_max=0., ota_pn_min=1., ota_np_max=0., ota_np_min=1., ota_nn_max=0., ota_nn_min=1.;
        float pratio, nratio;
        #endif
        #ifdef USE_ACTIVE
        float acons_sum1=0., acons_mean1=0., acons_var1=0., acons_max1=-INFINITY, acons_min1=INFINITY, acons_sum2=0., acons_mean2=0., acons_var2=0., acons_max2=-INFINITY, acons_min2=INFINITY;
        float acons_sum3=0., acons_mean3=0., acons_var3=0., acons_max3=-INFINITY, acons_min3=INFINITY, acons_sum4=0., acons_mean4=0., acons_var4=0., acons_max4=-INFINITY, acons_min4=INFINITY;
        float acons_nb1=0., acons_nb2=0., acons_nb3=0., acons_nb4=0.;
        int active_count = 0;
        int neighbor_row_index;
        float abs_coef, value;
        #endif

        for(int neighbor_index=0; neighbor_index < nb_neighbors; ++neighbor_index)
        {
            #ifdef USE_CDEG
            float cdeg = (float) SCIProwGetNLPNonz(neighbors[neighbor_index]);
            #ifdef USE_CDEG_MEAN
            cdeg_mean += cdeg;
            #endif
            #ifdef USE_CDEG_MAX
            cdeg_max = MAX(cdeg_max, cdeg);
            #endif
            #ifdef USE_CDEG_MIN
            cdeg_min = MIN(cdeg_min, cdeg);
            #endif
            #ifdef CDEG_VAR
            cdeg_var += cdeg * cdeg;
            #endif
            #endif

            #ifdef USE_RHS
            coef = (float) nonzero_coefs_raw[neighbor_index];
            lhs = (float) SCIProwGetLhs(neighbors[neighbor_index]);
            rhs = (float) SCIProwGetRhs(neighbors[neighbor_index]);
            if(!SCIPisInfinity(scip, REALABS(rhs)))
            {
                float value = (coef>0.) ? (coef / (REALABS(coef) + REALABS(rhs))):(0.);
                if(rhs >= 0.)
                {
                    #ifdef PRHS_RATIO_MAX
                    prhs_ratio_max = MAX(prhs_ratio_max, value);
                    #endif
                    #ifdef PRHS_RATIO_MIN
                    prhs_ratio_min = MIN(prhs_ratio_min, value);
                    #endif
                }
                else
                {
                    #ifdef NRHS_RATIO_MAX
                    nrhs_ratio_max = MAX(nrhs_ratio_max, value);
                    #endif
                    #ifdef NRHS_RATIO_MIN
                    nrhs_ratio_min = MIN(nrhs_ratio_min, value);
                    #endif
                }
            }
            if(!SCIPisInfinity(scip, REALABS(lhs)))
            {
                float value = (coef>0.) ? (coef / (REALABS(coef) + REALABS(lhs))):(0.);
                if(lhs <= 0.)
                {
                    #ifdef PRHS_RATIO_MAX
                    prhs_ratio_max = MAX(prhs_ratio_max, value);
                    #endif
                    #ifdef PRHS_RATIO_MIN
                    prhs_ratio_min = MIN(prhs_ratio_min, value);
                    #endif
                }
                else
                {
                    #ifdef NRHS_RATIO_MAX
                    nrhs_ratio_max = MAX(nrhs_ratio_max, value);
                    #endif
                    #ifdef NRHS_RATIO_MIN
                    nrhs_ratio_min = MIN(nrhs_ratio_min, value);
                    #endif
                }
            }
            #endif

            #ifdef USE_OTA
            SCIP_Real* all_coefs_raw = SCIProwGetVals(neighbors[neighbor_index]);
            int neighbor_ncolumns = SCIProwGetNNonz(neighbors[neighbor_index]);
            float pos_coef_sum=0., neg_coef_sum=0.;
            for(int neighbor_column_index=0; neighbor_column_index < neighbor_ncolumns; ++neighbor_column_index)
            {
                SCIP_Real neighbor_coef = all_coefs_raw[neighbor_column_index];
                if(neighbor_coef > 0) pos_coef_sum += neighbor_coef;
                else neg_coef_sum += neighbor_coef;
            }
            coef = (float) nonzero_coefs_raw[neighbor_index];
            if(coef > 0)
            {   
                #ifdef USE_OTA_P
                pratio = coef / pos_coef_sum;
                nratio = coef / (coef - neg_coef_sum);
                #ifdef OTA_PP_MAX
                ota_pp_max = MAX(ota_pp_max, pratio);
                #endif
                #ifdef OTA_PP_MIN
                ota_pp_min = MIN(ota_pp_min, pratio);
                #endif
                #ifdef OTA_PN_MAX
                ota_pn_max = MAX(ota_pn_max, nratio);
                #endif
                #ifdef OTA_PN_MIN
                ota_pn_min = MIN(ota_pn_min, nratio);
                #endif
                #endif
            }
            else if(coef < 0)
            {
                #ifdef USE_OTA_N
                pratio = coef / (coef - pos_coef_sum);
                nratio = coef / neg_coef_sum;
                #ifdef OTA_NP_MAX
                ota_np_max = MAX(ota_np_max, pratio);
                #endif
                #ifdef OTA_NP_MIN
                ota_np_min = MIN(ota_np_min, pratio);
                #endif
                #ifdef OTA_NN_MAX
                ota_nn_max = MAX(ota_nn_max, nratio);
                #endif
                #ifdef OTA_NN_MIN
                ota_nn_min = MIN(ota_nn_min, nratio);
                #endif
                #endif
            }
            #endif

            #ifdef USE_ACTIVE
            lhs = (float) SCIProwGetLhs(neighbors[neighbor_index]);
            rhs = (float) SCIProwGetRhs(neighbors[neighbor_index]);
            float activity = SCIPgetRowActivity(scip, neighbors[neighbor_index]);
            if( SCIPisEQ(scip, activity, rhs) || SCIPisEQ(scip, activity, lhs) )
            {
                active_count += 1.;
                neighbor_row_index = SCIProwGetLPPos(neighbors[neighbor_index]);
                abs_coef = REALABS(nonzero_coefs_raw[neighbor_index]);

                #ifdef USE_ACTIVE1
                value = act_cons_w1[neighbor_row_index] * abs_coef;
                #ifdef ACONS_NB1
                acons_nb1 += act_cons_w1[neighbor_row_index];
                #endif
                #ifdef USE_ACONS_SUM1
                acons_sum1 += value;
                #endif
                #ifdef ACONS_VAR1
                acons_var1 += value * value;
                #endif
                #ifdef ACONS_MAX1
                acons_max1 = MAX(acons_max1, value);
                #endif
                #ifdef ACONS_MIN1
                acons_min1 = MIN(acons_min1, value);
                #endif
                #endif

                #ifdef USE_ACTIVE2
                value = act_cons_w2[neighbor_row_index] * abs_coef;
                #ifdef ACONS_NB2
                acons_nb2 += act_cons_w2[neighbor_row_index];
                #endif
                #ifdef USE_ACONS_SUM2
                acons_sum2 += value;
                #endif
                #ifdef ACONS_VAR2
                acons_var2 += value * value;
                #endif
                #ifdef ACONS_MAX2
                acons_max2 = MAX(acons_max2, value);
                #endif
                #ifdef ACONS_MIN2
                acons_min2 = MIN(acons_min2, value);
                #endif
                #endif

                #ifdef USE_ACTIVE3
                value = act_cons_w3[neighbor_row_index] * abs_coef;
                #ifdef ACONS_NB3
                acons_nb3 += act_cons_w3[neighbor_row_index];
                #endif
                #ifdef USE_ACONS_SUM3
                acons_sum3 += value;
                #endif
                #ifdef ACONS_VAR3
                acons_var3 += value * value;
                #endif
                #ifdef ACONS_MAX3
                acons_max3 = MAX(acons_max3, value);
                #endif
                #ifdef ACONS_MIN3
                acons_min3 = MIN(acons_min3, value);
                #endif
                #endif

                #ifdef USE_ACTIVE4
                value = act_cons_w4[neighbor_row_index] * abs_coef;
                #ifdef ACONS_NB4
                acons_nb4 += act_cons_w4[neighbor_row_index];
                #endif
                #ifdef USE_ACONS_SUM4
                acons_sum4 += value;
                #endif
                #ifdef ACONS_VAR4
                acons_var4 += value * value;
                #endif
                #ifdef ACONS_MAX4
                acons_max4 = MAX(acons_max4, value);
                #endif
                #ifdef ACONS_MIN4
                acons_min4 = MIN(acons_min4, value);
                #endif
                #endif

            }
            #endif
        }
        #ifdef USE_CDEG
        if(nb_neighbors>0)
        {
            #ifdef USE_CDEG_MEAN
            cdeg_mean /= nb_neighbors;
            #endif
            #ifdef CDEG_VAR
            cdeg_var = cdeg_var/nb_neighbors - cdeg_mean * cdeg_mean;
            #endif
        }
        else cdeg_max = cdeg_min = 0.;

        #ifdef CDEG_MEAN
        allfeatures[c][CDEG_MEAN] = cdeg_mean;
        #endif
        #ifdef CDEG_MEAN_RATIO
        allfeatures[c][CDEG_MEAN_RATIO] = (cdeg_mean > 0.) ? (cdeg_mean / (rootinfo->root_cdeg_mean[col_i] + cdeg_mean)):(0.);
        #endif
        #ifdef CDEG_VAR
        allfeatures[c][CDEG_VAR] = cdeg_var;
        #endif
        #ifdef CDEG_MAX
        allfeatures[c][CDEG_MAX] = cdeg_max;
        #endif
        #ifdef CDEG_MAX_RATIO
        allfeatures[c][CDEG_MAX_RATIO] =  (cdeg_max > 0.) ? (cdeg_max / (rootinfo->root_cdeg_max[col_i] + cdeg_max)):(0.);
        #endif
        #ifdef CDEG_MIN
        allfeatures[c][CDEG_MIN] = cdeg_min;
        #endif
        #ifdef CDEG_MIN_RATIO
        allfeatures[c][CDEG_MIN_RATIO] =  (cdeg_min > 0.) ? (cdeg_min / (rootinfo->root_cdeg_min[col_i] + cdeg_min)):(0.);
        #endif
        #endif

        #ifdef USE_RHS
        #ifdef PRHS_RATIO_MAX
        allfeatures[c][PRHS_RATIO_MAX] = prhs_ratio_max;
        #endif
        #ifdef PRHS_RATIO_MIN
        allfeatures[c][PRHS_RATIO_MIN] = prhs_ratio_min;
        #endif
        #ifdef NRHS_RATIO_MAX
        allfeatures[c][NRHS_RATIO_MAX] = nrhs_ratio_max;
        #endif
        #ifdef NRHS_RATIO_MIN
        allfeatures[c][NRHS_RATIO_MIN] = nrhs_ratio_min;
        #endif
        #endif

        #ifdef USE_OTA
        #ifdef OTA_PP_MAX
        allfeatures[c][OTA_PP_MAX] = ota_pp_max;
        #endif
        #ifdef OTA_PP_MIN
        allfeatures[c][OTA_PP_MIN] = ota_pp_min;
        #endif
        #ifdef OTA_PN_MAX
        allfeatures[c][OTA_PN_MAX] = ota_pn_max;
        #endif
        #ifdef OTA_PN_MIN
        allfeatures[c][OTA_PN_MIN] = ota_pn_min;
        #endif
        #ifdef OTA_NP_MAX
        allfeatures[c][OTA_NP_MAX] = ota_np_max;
        #endif
        #ifdef OTA_NP_MIN
        allfeatures[c][OTA_NP_MIN] = ota_np_min;
        #endif
        #ifdef OTA_NN_MAX
        allfeatures[c][OTA_NN_MAX] = ota_nn_max;
        #endif
        #ifdef OTA_NN_MIN
        allfeatures[c][OTA_NN_MIN] = ota_nn_min;
        #endif
        #endif

        #ifdef USE_ACTIVE
        if(active_count > 0)
        {
            #ifdef USE_ACONS_MEAN1
            acons_mean1 = acons_sum1 / active_count;
            #endif
            #ifdef USE_ACONS_MEAN2
            acons_mean2 = acons_sum2 / active_count;
            #endif
            #ifdef USE_ACONS_MEAN3
            acons_mean3 = acons_sum3 / active_count;
            #endif
            #ifdef USE_ACONS_MEAN4
            acons_mean4 = acons_sum4 / active_count;
            #endif

            #ifdef ACONS_VAR1
            acons_var1 = acons_var1 / active_count - acons_mean1 * acons_mean1;
            #endif
            #ifdef ACONS_VAR2
            acons_var2 = acons_var2 / active_count - acons_mean2 * acons_mean2;
            #endif
            #ifdef ACONS_VAR3
            acons_var3 = acons_var3 / active_count - acons_mean3 * acons_mean3;
            #endif
            #ifdef ACONS_VAR4
            acons_var4 = acons_var4 / active_count - acons_mean4 * acons_mean4;
            #endif
        }
        else
        {
            acons_max1 = acons_min1 = acons_max2 = acons_min2 = acons_max3 = acons_min3 = acons_max4 = acons_min4 = 0.;
        }


        #ifdef ACONS_SUM1
        allfeatures[c][ACONS_SUM1] = acons_sum1;
        #endif
        #ifdef ACONS_SUM2
        allfeatures[c][ACONS_SUM2] = acons_sum2;
        #endif
        #ifdef ACONS_SUM3
        allfeatures[c][ACONS_SUM3] = acons_sum3;
        #endif
        #ifdef ACONS_SUM4
        allfeatures[c][ACONS_SUM4] = acons_sum4;
        #endif

        #ifdef ACONS_MEAN1
        allfeatures[c][ACONS_MEAN1] = acons_mean1;
        #endif
        #ifdef ACONS_MEAN2
        allfeatures[c][ACONS_MEAN2] = acons_mean2;
        #endif
        #ifdef ACONS_MEAN3
        allfeatures[c][ACONS_MEAN3] = acons_mean3;
        #endif
        #ifdef ACONS_MEAN4
        allfeatures[c][ACONS_MEAN4] = acons_mean4;
        #endif

        #ifdef ACONS_MAX1
        allfeatures[c][ACONS_MAX1] = acons_max1;
        #endif
        #ifdef ACONS_MAX2
        allfeatures[c][ACONS_MAX2] = acons_max2;
        #endif
        #ifdef ACONS_MAX3
        allfeatures[c][ACONS_MAX3] = acons_max3;
        #endif
        #ifdef ACONS_MAX4
        allfeatures[c][ACONS_MAX4] = acons_max4;
        #endif

        #ifdef ACONS_MIN1
        allfeatures[c][ACONS_MIN1] = acons_min1;
        #endif
        #ifdef ACONS_MIN2
        allfeatures[c][ACONS_MIN2] = acons_min2;
        #endif
        #ifdef ACONS_MIN3
        allfeatures[c][ACONS_MIN3] = acons_min3;
        #endif
        #ifdef ACONS_MIN4
        allfeatures[c][ACONS_MIN4] = acons_min4;
        #endif

        #ifdef ACONS_VAR1
        allfeatures[c][ACONS_VAR1] = acons_var1;
        #endif
        #ifdef ACONS_VAR2
        allfeatures[c][ACONS_VAR2] = acons_var2;
        #endif
        #ifdef ACONS_VAR3
        allfeatures[c][ACONS_VAR3] = acons_var3;
        #endif
        #ifdef ACONS_VAR4
        allfeatures[c][ACONS_VAR4] = acons_var4;
        #endif

        #ifdef ACONS_NB1
        allfeatures[c][ACONS_NB1] = acons_nb1;
        #endif
        #ifdef ACONS_NB2
        allfeatures[c][ACONS_NB2] = acons_nb2;
        #endif
        #ifdef ACONS_NB3
        allfeatures[c][ACONS_NB3] = acons_nb3;
        #endif
        #ifdef ACONS_NB4
        allfeatures[c][ACONS_NB4] = acons_nb4;
        #endif

        #endif

        #endif
    }

    #ifdef USE_ACTIVE1
    free(act_cons_w1);
    #endif
    #ifdef USE_ACTIVE2
    free(act_cons_w2);
    #endif
    #ifdef USE_ACTIVE3
    free(act_cons_w3);
    #endif
    #ifdef USE_ACTIVE4
    free(act_cons_w4);
    #endif

    #ifdef DEBUG_RAW
    printf("This is the raw features:\n");
    for(int i = 0; i < nlpcands; ++i)
    {
        for(int j=0; j < NUM_FEATURE; ++j)
            printf("%.5f ", allfeatures[i][j]);
        printf("\n");
    }
    printf("\n\n\n");
    #endif

    for(int j=0; j < NUM_FEATURE; ++j)
    {
        float min_val=INFINITY, max_val=-INFINITY, delta_val, temp_val;
        for(int i = 0; i < nlpcands; ++i)
        {
            temp_val = allfeatures[i][j];
            if(temp_val > max_val) max_val = temp_val;
            if(temp_val < min_val) min_val = temp_val;
        }
        delta_val = max_val - min_val;
        if(SCIPisZero(scip, delta_val)) delta_val = 1.;

        for(int i = 0; i < nlpcands; ++i)
            allfeatures[i][j] = (allfeatures[i][j] - min_val) / delta_val;
    }

    #ifdef DEBUG_NORMED
    printf("This is the normed features:\n");
    for(int i = 0; i < nlpcands; ++i)
    {
        for(int j=0; j < NUM_FEATURE; ++j)
            printf("%.5f ", allfeatures[i][j]);
        printf("\n");
    }
    printf("\n\n\n");
    #endif


    return SCIP_OKAY;
}

int symbGetBestCand(Symb_Feature* allfeatures, int nlpcands)
{
    int bestcand = -1;
    float bestscore = (float) -INFINITY;
    // SCIP_Real bestrootdiff = 0.0;

    #ifdef DEBUG_SCORE
    printf("This is the scores:\n");
    #endif

    for(int c = 0; c < nlpcands; ++c )
    {
        float score = ScoreFunc(allfeatures[c]);
        #ifdef DEBUG_SCORE
        printf("%.5f\t", score);
        #endif
        // SCIP_Real rootdiff = REALABS(solval - SCIPvarGetRootSol(lpcands[c]));
        //  || (SCIPisSumEQ(scip, score, bestscore) && rootdiff > bestrootdiff)
        // if( SCIPisSumGT(scip, score, bestscore) )
        if(score>bestscore)
        {
            bestcand = c;
            bestscore = score;
            // bestrootdiff = rootdiff;
        }
    }
    // assert(0 <= bestcand && bestcand < nlpcands);
    // assert(!SCIPisFeasIntegral(scip, lpcandssol[bestcand]));
    // assert(!SCIPisFeasIntegral(scip, SCIPvarGetSol(lpcands[bestcand], TRUE)));

    #ifdef DEBUG_SCORE
    printf("\n\n\n");
    #endif

    return bestcand;
}

SCIP_RETCODE getSymbGoodCands(Symb_Feature* allfeatures, SCIP_VAR** lpcands, int nlpcands, SCIP_VAR** goodcands, int nsymbcands, SCIP_BRANCHRULEDATA* branchruledata)
{
    int* candsindices = branchruledata->tempinfo.candsindices;
    float* scoreslist = branchruledata->tempinfo.scoreslist;

    // SCIP_ALLOC( BMSallocMemorySize(&candsindices, sizeof(int) * nsymbcands ) );
    // SCIP_ALLOC( BMSallocMemorySize(&scoreslist, sizeof(float) * nsymbcands ) );
    assert(candsindices!= NULL && scoreslist != NULL);

    for(int i = 0; i < nsymbcands; ++i) scoreslist[i] = -INFINITY;

    for(int c = 0; c < nlpcands; ++c )
    {
        float score = ScoreFunc(allfeatures[c]);
        #ifdef DEBUG_SCORE
        printf("%.5f\t", score);
        #endif

        if(score > scoreslist[0])
        {
            int j;
            for(j=1; j < nsymbcands && score > scoreslist[j]; ++j)
            {
                scoreslist[j-1] = scoreslist[j];
                candsindices[j-1] = candsindices[j];
            }
            scoreslist[j-1] = score;
            candsindices[j-1] = c;
        }
    }


    #ifdef DEBUG_SCORE
    for(int i = 0; i < nsymbcands; ++i) assert(candsindices[i] >= 0 && candsindices[i] < nlpcands && scoreslist[i] > -INFINITY);
    #endif

    for(int i = 0; i < nsymbcands; ++i) goodcands[i] = lpcands[candsindices[i]];

    // free(candsindices);
    // free(scoreslist);

    return SCIP_OKAY;
}


SCIP_RETCODE allocAndGetSymbGoodCands(SCIP* scip, SCIP_BRANCHRULE* branchrule, SCIP_VAR*** goodcands_ptr, int* ngoodcands)
{
    int nsymbcands;
    SCIP_BRANCHRULEDATA* branchruledata;
    
    branchruledata = SCIPbranchruleGetData(branchrule);

    if (branchruledata->tempinfo.candsindices != NULL) ;
    else
    {
        branchruledata->tempinfo.goodcandslen = ceil(((float) SCIPgetNVars(scip)) * branchruledata->bestratio) * 10;
        assert(branchruledata->tempinfo.goodcandslen > 0);

        SCIP_ALLOC( BMSallocMemorySize(&(branchruledata->tempinfo.candsindices), branchruledata->tempinfo.goodcandslen * sizeof(int)) );
        SCIP_ALLOC( BMSallocMemorySize(&(branchruledata->tempinfo.scoreslist), branchruledata->tempinfo.goodcandslen * sizeof(float)) );
    }


    clock_t start_time = clock();
    Symb_Feature* allfeatures;
    int nlpcands;
    SCIP_VAR** lpcands;

    SCIP_CALL( symbAllocSetFeatures(&allfeatures, &lpcands, &nlpcands, scip, branchrule) );
    // xxx
    nsymbcands = ceil( ((float)nlpcands) *  branchruledata->bestratio );
    *ngoodcands = nsymbcands;

    if(nsymbcands > 0)
    {
        SCIP_ALLOC( BMSallocMemorySize(goodcands_ptr, sizeof(SCIP_VAR*) * nsymbcands ) );
        assert(*goodcands_ptr != NULL);
        SCIP_CALL( getSymbGoodCands(allfeatures, lpcands, nlpcands, *goodcands_ptr, nsymbcands, branchruledata) );
    }
    else *goodcands_ptr=NULL;

    branchruledata->decisiontime += (SCIP_Real) (clock() - start_time) / CLOCKS_PER_SEC;
    free(allfeatures);

    return SCIP_OKAY;
}


SCIP_DECL_BRANCHEXECLP(branchExeclpSymb)
{
    SCIP_BRANCHRULEDATA* branchruledata;
    branchruledata = SCIPbranchruleGetData(branchrule);

    clock_t start_time = clock();
    Symb_Feature* allfeatures;
    int nlpcands;
    SCIP_VAR** lpcands;

    SCIP_CALL( symbAllocSetFeatures(&allfeatures, &lpcands, &nlpcands, scip, branchrule) );
    int bestcand = symbGetBestCand(allfeatures, nlpcands);

    branchruledata->decisiontime += (SCIP_Real) (clock() - start_time) / CLOCKS_PER_SEC;

    // SCIPdebugMsg(scip, " -> %d cands, selected cand %d: variable <%s> (solval=%g, score=%g)\n",
    //     nlpcands, bestcand, SCIPvarGetName(lpcands[bestcand]), lpcandssol[bestcand], bestscore);

    /* perform the branching */
    SCIP_CALL( SCIPbranchVar(scip, lpcands[bestcand], NULL, NULL, NULL) );

    free(allfeatures);
    *result = SCIP_BRANCHED;

    return SCIP_OKAY;
}




float ScoreFunc(Symb_Feature feature)
{return 0;}