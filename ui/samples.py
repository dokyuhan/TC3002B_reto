"""Static sanitized Java demo pairs for the Streamlit dashboard.

All fixtures are embedded as static Python strings derived from ConPlag
and IR Plag benchmarks so the UI never reads raw dataset files at runtime.
"""

from __future__ import annotations

DEMO_SAMPLES: dict[str, dict[str, str]] = {
    "ConPlag: Plagiarized (sample)": {
        "title": "ConPlag: Plagiarized (sample)",
        "description": (
            "ConPlag Java pair. Both files run the same minimum sweep with "
            "renamed variables and a changed loop."
        ),
        "expected_label": "Plagiarized",
        "language": "java",
        "code_a": """import java.util.*;

public class ContestA {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int t = scanner.nextInt();
        int inf = 1000300300;
        while (t-- > 0) {
            int n = scanner.nextInt();
            int k = scanner.nextInt();
            int[] c = new int[n];
            Arrays.fill(c, inf);
            int[] a = new int[k];
            int[] b = new int[k];
            for (int i = 0; i < k; ++i) a[i] = scanner.nextInt() - 1;
            for (int i = 0; i < k; ++i) b[i] = scanner.nextInt();
            for (int i = 0; i < k; ++i) c[a[i]] = b[i];
            for (int i = 1; i < n; ++i) {
                c[i] = Math.min(c[i], c[i - 1] + 1);
            }
            for (int i = n - 2; i >= 0; --i) {
                c[i] = Math.min(c[i], c[i + 1] + 1);
            }
            for (int i = 0; i < n; ++i) System.out.print(c[i] + " ");
            System.out.println();
        }
    }
}
""",
        "code_b": """import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.StringTokenizer;

public class Main {
    public static void main(String[] args) {
        FastReader s = new FastReader();
        int t = 1;
        t = s.ni();
        for (int test = 1; test <= t; test++) {
            int n = s.ni(), k = s.ni();
            int position[] = s.readArray(k), temp[] = s.readArray(k);
            int ans[] = new int[n];
            Arrays.fill(ans, Integer.MAX_VALUE / 2);
            for (int i = 0; i < k; i++) {
                ans[position[i] - 1] = temp[i];
            }
            for (int i = 1; i < n; i++) {
                ans[i] = Math.min(ans[i - 1] + 1, ans[i]);
            }
            for (int i = n - 2; i >= 0; i--) {
                ans[i] = Math.min(ans[i], ans[i + 1] + 1);
            }
            for (int i = 0; i < n; i++)
                System.out.print(ans[i] + " ");
            System.out.println();
        }
    }
}
""",
    },
    "ConPlag: Clean (sample)": {
        "title": "ConPlag: Clean (sample)",
        "description": (
            "ConPlag Java pair. Both files solve the same dynamic programming "
            "task with different state logic."
        ),
        "expected_label": "Clean",
        "language": "java",
        "code_a": """import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

public class E1525D {
    public static void main(String[] args) {
        Scanner scn = new Scanner(System.in);
        int n = scn.nextInt();
        int[] arr = new int[n];
        ArrayList<Integer> occupied = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            arr[i] = scn.nextInt();
            if (arr[i] == 1) occupied.add(i);
        }
        int[][] dp = new int[n + 1][occupied.size() + 1];
        for (int[] row : dp) Arrays.fill(row, (int) 1e9);
        dp[0][0] = 0;
        for (int i = 1; i <= n; i++) {
            dp[i][0] = 0;
            for (int j = 1; j <= occupied.size(); j++) {
                dp[i][j] = dp[i - 1][j];
                if (arr[i - 1] == 0) {
                    dp[i][j] = Math.min(
                        dp[i][j],
                        dp[i - 1][j - 1] + Math.abs(i - 1 - occupied.get(j - 1))
                    );
                }
            }
        }
        System.out.println(dp[n][occupied.size()]);
    }
}
""",
        "code_b": """import java.util.ArrayList;
import java.util.Scanner;

public class Armchair {
    public static void main(String args[]) {
        Scanner in = new Scanner(System.in);
        int n = in.nextInt();
        ArrayList<Integer> list1 = new ArrayList<Integer>();
        ArrayList<Integer> list2 = new ArrayList<Integer>();
        for (int i = 0; i < n; i++) {
            int a = in.nextInt();
            if (a == 0)
                list2.add(i);
            else
                list1.add(i);
        }
        long dp[][] = new long[list1.size() + 1][list2.size() + 1];

        solve(list1, list2, dp);
        System.out.println(dp[list1.size()][list2.size()]);
    }

    public static void solve(
        ArrayList<Integer> list1,
        ArrayList<Integer> list2,
        long dp[][]
    ) {
        for (int i = 1; i <= list1.size(); i++)
            dp[i][0] = Integer.MAX_VALUE;

        for (int i = 1; i <= list1.size(); i++) {
            for (int j = 1; j <= list2.size(); j++) {
                dp[i][j] = Math.min(
                    Math.abs(list1.get(i - 1) - list2.get(j - 1)) + dp[i - 1][j - 1],
                    dp[i][j - 1]
                );
            }
        }
    }
}
""",
    },
    "IR Plag: L6 Obfuscation (sample)": {
        "title": "IR Plag: L6 Obfuscation (sample)",
        "description": (
            "IR Plag Java pair (case 07, level L6). One file changes loop "
            "types, traversal order, methods, and names."
        ),
        "expected_label": "Plagiarized",
        "language": "java",
        "code_a": """import java.util.Scanner;

public class T7 {
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);

        System.out.print("Enter a 4 by 4 matrix row by row: ");
        double[][] m = new double[4][4];

        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                m[i][j] = input.nextDouble();

        System.out.print(
            "Sum of the elements in the major diagonal is " + sumMajorDiagonal(m)
        );
    }

    public static double sumMajorDiagonal(double[][] m) {
        double sum = 0;

        for (int i = 0; i < m.length; i++)
            sum += m[i][i];

        return sum;
    }
}
""",
        "code_b": """import java.util.*;

public class L6 {
    public static double jumlahDiagonal(double[][] m) {
        double sum = 0;
        int hit = m.length - 1;

        while (hit >= 0) {
            sum += m[hit][hit];
            hit--;
        }
        return sum;
    }

    public static void print(double[][] m) {
        System.out.print(
            "Sum of the elements in the major diagonal is " + jumlahDiagonal(m)
        );
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        double[][] m = new double[4][4];
        System.out.println("Enter a 4 by 4 matrix row by row: ");

        for (int i = 3; i >= 0; i--) {
            for (int j = 3; j >= 0; j--) {
                m[i][j] = sc.nextDouble();
            }
        }
        print(m);
    }
}
""",
    },
}
